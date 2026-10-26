import re
import json
import uuid
import sqlite3
import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

load_dotenv()

SKIP_COLUMNS = ["created_timestamp", "updated_timestamp", "is_active"]
MODEL_NAME = "openai/gpt-oss-120b"


def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def load_llm():
    return ChatGroq(model=MODEL_NAME, temperature=0)


def row_to_text(table_name, row):
    parts = [f"table: {table_name}"]
    for column, value in row.items():
        if column in SKIP_COLUMNS:
            continue
        parts.append(f"{column}: {value}")
    return " | ".join(parts)


def make_documents(dataframes, skip_tables):
    documents = []
    for name, df in dataframes.items():
        if name in skip_tables:
            continue
        for index, row in df.iterrows():
            documents.append(Document(
                page_content=row_to_text(name, row),
                metadata={"table": name, "row": int(index)}
            ))
    return documents


def get_schema(conn):
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)["name"].tolist()
    schema = ""
    for table in tables:
        schema += f"\nTable: {table}\n"
        cols = pd.read_sql(f"SELECT name, type FROM pragma_table_info('{table}')", conn)
        for _, col in cols.iterrows():
            line = f"  - {col['name']} ({col['type']})"
            if col["type"] == "TEXT":
                values = pd.read_sql(
                    f'SELECT DISTINCT "{col["name"]}" FROM "{table}" LIMIT 11', conn
                ).iloc[:, 0].tolist()
                if len(values) <= 10:
                    line += f" values: {values}"
            schema += line + "\n"
    return schema


def build_engine(dataframes, embedding_model, skip_tables=None):
    skip_tables = skip_tables or []

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    for name, df in dataframes.items():
        df.to_sql(name, conn, index=False, if_exists="replace")
    schema = get_schema(conn)

    vector_store = Chroma(
        collection_name=f"data_{uuid.uuid4().hex}",
        embedding_function=embedding_model
    )
    documents = make_documents(dataframes, skip_tables)
    for i in range(0, len(documents), 500):
        vector_store.add_documents(documents[i : i + 500])

    rag_tables = [t for t in dataframes if t not in skip_tables]

    return {
        "conn": conn,
        "schema": schema,
        "vector_store": vector_store,
        "rag_tables": rag_tables,
    }

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful data assistant. "
     "Answer using ONLY the data provided. "
     "If the answer is not in the data, say \"I don't know based on the data.\" "
     "Mention IDs and names when relevant."),
    ("human", "Data:\n{context}\n\nQuestion: {question}")
])

SQL_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert at writing SQLite queries.\n"
     "Given the database schema below, write ONE SQLite query that answers the user's question.\n"
     "Rules:\n"
     "- Use only the tables and columns in the schema.\n"
     "- Use SQLite syntax.\n"
     "- Only write SELECT queries. Never write INSERT, UPDATE, DELETE, DROP or ALTER.\n"
     "- Do not add extra filters (such as is_active) unless the user asks for them.\n"
     "- For totals, counts or averages per group, sort the result by that value from highest to lowest, unless the user asks for a different order.\n"
     "- Return ONLY the SQL query. No explanation, no markdown.\n\n"
     "Schema:\n{schema}"),
    ("human", "{question}")
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful data assistant. "
     "Answer the user's question in clear, simple English using ONLY the SQL result provided.\n"
     "The full result table is shown to the user separately, right below your answer.\n"
     "- Do NOT repeat the result as a table or code block. Mention only the most important values in your sentences.\n"
     "- If the result has many rows, give a short summary, such as how many rows there are and a few key highlights from the top rows.\n"
     "- Format large numbers with commas.\n"
     "- If the result is empty, say that no matching data was found."),
    ("human", "Question: {question}\n\nSQL used:\n{sql}\n\nTotal rows: {total_rows}\n\nResult:\n{result}")
])

SCHEMA_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful data assistant. "
     "Answer the user's question about the structure of their data using ONLY the schema below. "
     "Use simple English and short lists.\n\n"
     "Schema:\n{schema}"),
    ("human", "{question}")
])

ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a router for a data chatbot. Decide how to answer the user's question.\n"
     "Choose 'schema' if the question is about the structure of the data itself, "
     "such as which tables or columns exist, or what kind of data is available.\n"
     "Choose 'sql' if the question needs exact values, counting, totals, averages, max or min, "
     "rankings, listing ALL matching rows, specific names or IDs, or combining tables.\n"
     "Choose 'rag' if the question is fuzzy or descriptive, about meaning or similarity, "
     "such as recommendations or 'find something like X'.\n"
     "If you choose 'rag', also pick the single most relevant table from this list: {tables}\n"
     "Respond ONLY with JSON, for example: {{\"route\": \"rag\", \"table\": \"products\"}}"),
    ("human", "{question}")
])


def ask_rag(engine, llm, question, table=None, k=10):
    search_filter = {"table": table} if table else None
    docs = engine["vector_store"].similarity_search(question, k=k, filter=search_filter)
    context = "\n".join(doc.page_content for doc in docs)
    messages = RAG_PROMPT.invoke({"context": context, "question": question})
    return llm.invoke(messages).content, docs


def ask_schema(engine, llm, question):
    messages = SCHEMA_PROMPT.invoke({"schema": engine["schema"], "question": question})
    return llm.invoke(messages).content


def write_sql(engine, llm, question):
    messages = SQL_PROMPT.invoke({"schema": engine["schema"], "question": question})
    sql = llm.invoke(messages).content.strip()
    return sql.replace("```sql", "").replace("```", "").strip()


def is_safe(sql):
    cleaned = sql.strip().lower()
    if not (cleaned.startswith("select") or cleaned.startswith("with")):
        return False
    blocked = ["insert", "update", "delete", "drop", "alter", "create", "attach", "pragma"]
    words = re.findall(r"[a-z_]+", cleaned)
    return not any(word in blocked for word in words)


def run_sql(engine, llm, question, max_tries=2):
    sql = write_sql(engine, llm, question)
    error = None
    for attempt in range(max_tries):
        if not is_safe(sql):
            return sql, None, "Blocked: only read-only SELECT queries are allowed."
        try:
            result = pd.read_sql(sql, engine["conn"])
            return sql, result, None
        except Exception as e:
            error = str(e)
            retry_question = (
                f"{question}\n\nYour previous query was:\n{sql}\n"
                f"It failed with this error: {error}\nWrite a corrected query."
            )
            sql = write_sql(engine, llm, retry_question)
    return sql, None, error


def ask_sql(engine, llm, question):
    sql, result, error = run_sql(engine, llm, question)
    if error:
        return f"Sorry, I couldn't answer that. ({error})", sql, None
    result_text = result.head(30).to_string(index=False)
    if len(result) > 30:
        result_text += f"\n... and {len(result) - 30} more rows"
    messages = ANSWER_PROMPT.invoke({
        "question": question,
        "sql": sql,
        "total_rows": len(result),
        "result": result_text,
    })
    return llm.invoke(messages).content, sql, result


def route_question(engine, llm, question):
    messages = ROUTER_PROMPT.invoke({"tables": ", ".join(engine["rag_tables"]), "question": question})
    text = llm.invoke(messages).content.strip().replace("```json", "").replace("```", "").strip()
    try:
        decision = json.loads(text)
    except json.JSONDecodeError:
        decision = {"route": "sql", "table": None}
    if decision.get("route") not in ("sql", "rag", "schema"):
        decision["route"] = "sql"
    if decision.get("table") not in engine["rag_tables"]:
        decision["table"] = None
    return decision


def chat(engine, llm, question):
    decision = route_question(engine, llm, question)

    if decision["route"] == "schema":
        answer = ask_schema(engine, llm, question)
        return {"answer": answer, "route": "schema", "sql": None, "table": None, "result": None}

    if decision["route"] == "sql":
        answer, sql, result = ask_sql(engine, llm, question)
        return {"answer": answer, "route": "sql", "sql": sql, "table": None, "result": result}

    answer, sources = ask_rag(engine, llm, question, table=decision["table"])
    return {"answer": answer, "route": "rag", "sql": None, "table": decision["table"], "result": None}