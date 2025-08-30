import streamlit as st
from utils import extract_text_from_pdf, chunk_text, format_context
from rag import RAGIndex
from ibm_llm import build_prompt, call_watsonx

st.set_page_config(page_title="StudyMate (MVP)", layout="wide")
st.title("StudyMate — PDF Q&A (MVP)")

with st.expander("How it works"):
    st.markdown(
        "- Upload one or more PDFs.\n"
        "- We extract text, chunk it, and build a semantic index.\n"
        "- Ask a question; we retrieve the most relevant chunks and (optionally) ask IBM Watsonx Mixtral to answer.\n"
        "- You'll always see the referenced chunks for transparency."
    )

if "rag" not in st.session_state:
    st.session_state.rag = RAGIndex()
if "history" not in st.session_state:
    st.session_state.history = []

uploaded = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
if uploaded:
    new_chunks = []
    for f in uploaded:
        txt = extract_text_from_pdf(f)
        chunks = chunk_text(txt, 500, 100)
        for c in chunks:
            new_chunks.append({"text": c, "source": f.name})
    if new_chunks:
        st.session_state.rag.add(new_chunks)
        st.success(f"Indexed {len(new_chunks)} chunks from {len(uploaded)} file(s).")

q = st.text_input("Ask a question about your PDFs")
go = st.button("Answer")

if go and q.strip():
    if not st.session_state.rag.is_ready():
        st.warning("Please upload at least one PDF first.")
    else:
        top = st.session_state.rag.search(q, k=3)
        ctx = format_context(top, max_chars=3000)
        prompt = build_prompt(ctx, q)
        answer = call_watsonx(prompt)
        if not answer:
            answer = "I couldn't reach the IBM model right now. Here are the most relevant excerpts from your PDFs:\n\n" + ctx
        st.markdown("### Answer")
        st.write(answer)
        with st.expander("Referenced Paragraphs"):
            for item in top:
                st.markdown(f"**{item['source']}** — score {item['score']:.3f}")
                st.write(item["text"])

        st.session_state.history.append({"q": q, "a": answer, "ctx": top})

if st.session_state.history:
    st.markdown("---")
    st.subheader("Q&A History (this session)")
    for i, h in enumerate(reversed(st.session_state.history), start=1):
        st.markdown(f"**Q{i}:** {h['q']}")
        st.markdown(f"**A{i}:** {h['a'][:500]}{'...' if len(h['a'])>500 else ''}")
