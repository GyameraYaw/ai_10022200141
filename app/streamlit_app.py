# CS4241 Manual RAG Chatbot — Streamlit UI
# Student: Yaw Acheampong Ahenkora Gyamera | Index: 10022200141

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src.config import INDEX_PATH

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Ghana RAG Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Ghana RAG Assistant")
    st.caption("Grounded answers · no hallucination")
    st.divider()

    st.subheader("Data Sources")
    st.markdown(
        "- 🗳️ Presidential Elections 1992–2020\n"
        "- 💰 Ghana 2025 Budget Statement"
    )
    st.divider()

    st.subheader("Pipeline Settings")
    prompt_version = st.selectbox(
        "Prompt template",
        ["v3_grounded_cite", "v2_grounded", "v1_naive"],
        index=0,
        help="v3 requires the LLM to cite every claim — recommended",
    )
    top_k = st.slider("Top-K chunks", min_value=1, max_value=10, value=5)
    use_memory = st.toggle("Session memory", value=True,
                           help="Resolves follow-up questions using conversation history")

    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.chat_history = []
        if "pipeline" in st.session_state:
            st.session_state.pipeline.memory.clear()
        st.rerun()

# ── Pipeline loader ───────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading RAG pipeline…")
def load_pipeline(prompt_ver: str, k: int, memory: bool):
    if not INDEX_PATH.exists():
        return None
    from src.rag_pipeline import RAGPipeline
    return RAGPipeline(prompt_version=prompt_ver, top_k=k, use_memory=memory)


if not INDEX_PATH.exists():
    st.error(
        "⚠️ Vector index not found. Please run:\n\n"
        "```bash\npython scripts/download_data.py\npython scripts/build_index.py\n```"
    )
    st.stop()

pipeline = load_pipeline(prompt_version, top_k, use_memory)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Header ────────────────────────────────────────────────────────────────────

st.title("🔍 Ghana RAG Assistant")
st.caption(
    "Ask about Ghana's presidential election results (1992–2020) or the 2025 national budget. "
    "Answers are grounded in the source documents."
)

if not st.session_state.chat_history:
    st.markdown(" ")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**Try:** How many votes did Akufo-Addo get in Ashanti in 2020?")
    with c2:
        st.info("**Try:** What is the 2025 budget allocation for health?")
    with c3:
        st.info("**Try:** What is Ghana's GDP growth target for 2025?")

# ── Intent label helper ───────────────────────────────────────────────────────

_INTENT_LABEL = {
    "election_numeric": "🗳️ Election",
    "budget_policy":    "💰 Budget",
    "general":          "🔍 General",
}

# ── Render chunk panel ────────────────────────────────────────────────────────

def render_chunks(retrieved: list, selected: list) -> None:
    selected_ids = {s["chunk_id"] for s in selected}
    for i, chunk in enumerate(retrieved, 1):
        source_label = "🗳️ Election" if chunk["source"] == "election_csv" else "💰 Budget"
        used = chunk["chunk_id"] in selected_ids
        badge = "✅ used" if used else "🔻 filtered"
        st.markdown(
            f"**Chunk {i}** — {source_label} — `{chunk['chunk_id']}` — {badge}  \n"
            f"Combined: `{chunk['combined_score']:.4f}` | "
            f"Vector: `{chunk['vector_score']:.4f}` | "
            f"BM25: `{chunk['bm25_score']:.4f}`"
        )
        st.text(chunk["text"][:300] + ("…" if len(chunk["text"]) > 300 else ""))
        st.divider()

# ── Render past turns ─────────────────────────────────────────────────────────

for turn in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(turn["query"])

    with st.chat_message("assistant"):
        st.caption(f"Intent: {_INTENT_LABEL.get(turn['intent'], turn['intent'])}")
        st.write(turn["answer"])

        with st.expander(f"📄 Retrieved chunks ({len(turn['retrieved'])} total, {len(turn['selected'])} used)"):
            render_chunks(turn["retrieved"], turn["selected"])

        with st.expander("📝 Prompt sent to LLM"):
            st.code(turn["prompt"], language="text")

        if turn.get("log_path"):
            log_file = Path(turn["log_path"])
            if log_file.exists():
                st.download_button(
                    "⬇️ Download run log",
                    data=log_file.read_text(encoding="utf-8"),
                    file_name=log_file.name,
                    mime="text/plain",
                    key=f"dl_{turn['timestamp']}",
                )

# ── Query input ───────────────────────────────────────────────────────────────

if user_query := st.chat_input("Ask about Ghana's elections or 2025 budget…"):
    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating answer…"):
            result = pipeline.answer(user_query)

        st.caption(f"Intent: {_INTENT_LABEL.get(result['intent'], result['intent'])}")

        if result["resolved_query"] != result["query"]:
            st.info(f"🔗 Query expanded to: *{result['resolved_query']}*")

        st.write(result["answer"])

        with st.expander(f"📄 Retrieved chunks ({len(result['retrieved'])} total, {len(result['selected'])} used)"):
            render_chunks(result["retrieved"], result["selected"])

        with st.expander("📝 Prompt sent to LLM"):
            st.code(result["prompt"], language="text")

        log_file = Path(result["log_path"])
        if log_file.exists():
            st.download_button(
                "⬇️ Download run log",
                data=log_file.read_text(encoding="utf-8"),
                file_name=log_file.name,
                mime="text/plain",
                key=f"dl_{result['timestamp']}",
            )

    st.session_state.chat_history.append(result)
