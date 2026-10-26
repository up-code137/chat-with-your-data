import re
import streamlit as st
import pandas as pd
from pathlib import Path
import engine

MAX_RAG_ROWS = 2000

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
    skipped = [name for name, df in dataframes.items() if len(df) > MAX_RAG_ROWS]
    with st.spinner(f"Preparing {len(dataframes)} table(s)... this can take a minute."):
        st.session_state.engine = engine.build_engine(dataframes, embedding_model, skip_tables=skipped)
    st.session_state.data_key = key
    st.session_state.tables = {name: len(df) for name, df in dataframes.items()}
    st.session_state.skipped = skipped
    st.session_state.messages = []


def friendly_error(e):
    text = str(e).lower()
    if "rate limit" in text or "429" in text:
        return "⏳ The AI service is busy (free usage limit reached). Please wait a minute and try again."
    if "api key" in text or "401" in text:
        return "🔑 There's a problem with the API key. Please check the Groq key setup."
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

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"].replace("$", "\\$"))
        if msg["role"] == "assistant":
            show_extras(msg, i)

question = st.chat_input("Ask a question about your data...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

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