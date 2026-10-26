import re
import time
import chromadb
import streamlit as st
import pandas as pd
from pathlib import Path
import engine

MAX_RAG_ROWS = 2000
MAX_QUESTIONS = 30
MAX_IDLE_SECONDS = 60 * 60

ROUTE_LABELS = {
    "sql": "SQL (exact calculation on all rows)",
    "rag": "RAG (search by meaning)",
    "schema": "Schema (structure of your data)",
}

st.set_page_config(page_title="Chat with Your Data", page_icon="💬")

st.title("💬 Chat with Your Data")
st.write("Upload CSV files and ask questions in plain English.")


@st.cache_resource
def get_models():
    return engine.load_embedding_model(), engine.load_llm()


@st.cache_resource
def get_chroma_client():
    return chromadb.EphemeralClient()


@st.cache_resource
def get_last_used():
    return {}


def cleanup_idle_collections():
    client = get_chroma_client()
    last_used = get_last_used()
    now = time.time()
    for name, used_at in list(last_used.items()):
        if now - used_at > MAX_IDLE_SECONDS:
            try:
                client.delete_collection(name)
            except Exception:
                pass
            last_used.pop(name, None)


def clean_name(filename):
    name = Path(filename).stem.lower()
    name = re.sub(r"[^a-z0-9_]", "_", name)
    if name[0].isdigit():
        name = "t_" + name
    return name


def read_uploads(uploaded_files):
    dataframes = {}
    for f in uploaded_files:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            st.sidebar.error(f"Could not read {f.name}: {e}")
            continue
        if df.empty:
            st.sidebar.warning(f"{f.name} has no rows, so it was skipped.")
            continue
        dataframes[clean_name(f.name)] = df
    return dataframes


def set_data(dataframes, key):
    client = get_chroma_client()
    last_used = get_last_used()
    cleanup_idle_collections()

    skipped = [name for name, df in dataframes.items() if len(df) > MAX_RAG_ROWS]
    with st.spinner(f"Preparing {len(dataframes)} table(s)... this can take a minute."):
        new_engine = engine.build_engine(
            dataframes, embedding_model, skip_tables=skipped, client=client
        )
    last_used[new_engine["collection"]] = time.time()

    old_engine = st.session_state.get("engine")

    st.session_state.data_key = key
    st.session_state.tables = {name: len(df) for name, df in dataframes.items()}
    st.session_state.skipped = skipped
    st.session_state.messages = []
    st.session_state.engine = new_engine

    if old_engine:
        try:
            client.delete_collection(old_engine["collection"])
        except Exception:
            pass
        last_used.pop(old_engine["collection"], None)


def friendly_error(e):
    text = str(e).lower()
    if "rate limit" in text or "429" in text:
        return "⏳ The AI service is busy (free usage limit reached). Please wait a minute and try again."
    if "api key" in text or "401" in text:
        return "🔑 There's a problem with the API key. Please check the Groq key setup."
    if "does not exist" in text:
        return "⌛ Your data was cleared after 1 hour of inactivity. Please upload your files (or load the sample data) again."
    return f"⚠️ Something went wrong: {e}"


def show_extras(msg, i):
    if msg.get("result") is not None and not msg["result"].empty:
        st.dataframe(msg["result"])
        st.download_button(
            "⬇️ Download as CSV",
            msg["result"].to_csv(index=False),
            file_name="result.csv",
            mime="text/csv",
            key=f"download_{i}",
        )
    if msg.get("route"):
        with st.expander("Details"):
            st.write(f"**Answered using:** {ROUTE_LABELS[msg['route']]}")
            if msg.get("table"):
                st.write(f"**Searched table:** {msg['table']}")
            if msg.get("sql"):
                st.code(msg["sql"], language="sql")


with st.spinner("Loading AI models..."):
    embedding_model, llm = get_models()

with st.sidebar:
    st.header("📂 Your data")
    uploaded_files = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)
    use_sample = st.button("Or try sample Walmart data")

if uploaded_files:
    key = tuple(sorted((f.name, f.size) for f in uploaded_files))
    if st.session_state.get("data_key") != key:
        dataframes = read_uploads(uploaded_files)
        if dataframes:
            set_data(dataframes, key)
elif use_sample and st.session_state.get("data_key") != "sample":
    dataframes = {f.stem: pd.read_csv(f) for f in Path("data").glob("*.csv")}
    set_data(dataframes, "sample")

if "engine" not in st.session_state:
    st.info("👈 Upload one or more CSV files, or try the sample data, to start chatting.")
    st.stop()

with st.sidebar:
    st.subheader("Loaded tables")
    for name, rows in st.session_state.tables.items():
        note = " (SQL only)" if name in st.session_state.skipped else ""
        st.write(f"✅ {name}: {rows:,} rows{note}")
    st.caption(f"🧪 Collections in memory: {len(get_chroma_client().list_collections())}")

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"].replace("$", "\\$"))
        if msg["role"] == "assistant":
            show_extras(msg, i)

used = st.session_state.get("question_count", 0)
limit_reached = used >= MAX_QUESTIONS
if limit_reached:
    st.warning(f"You've used all {MAX_QUESTIONS} questions for this session. Thanks for trying the app! 🙏")
question = st.chat_input("Ask a question about your data...", disabled=limit_reached)

if question:
    st.session_state.question_count = used + 1
    get_last_used()[st.session_state.engine["collection"]] = time.time()
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question.replace("$", "\\$"))

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                result = engine.chat(st.session_state.engine, llm, question)
            msg = {
                "role": "assistant",
                "content": result["answer"],
                "route": result["route"],
                "sql": result["sql"],
                "table": result["table"],
                "result": result["result"],
            }
        except Exception as e:
            msg = {"role": "assistant", "content": friendly_error(e)}
        st.markdown(msg["content"].replace("$", "\\$"))
        show_extras(msg, len(st.session_state.messages))

    st.session_state.messages.append(msg)