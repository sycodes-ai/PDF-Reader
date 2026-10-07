"""Chat with your PDFs: a small RAG app built with Streamlit, LangChain, OpenAI and Chroma.

Pipeline: upload PDFs -> extract text -> split into overlapping chunks ->
embed with OpenAI -> store in Chroma -> answer questions with a RetrievalQA chain.
"""

import os

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4


def get_pdf_text(pdf_docs):
    """Return the text of every page of every uploaded PDF."""
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            # extract_text() can return None for pages without a text layer
            text += (page.extract_text() or "") + "\n"
    return text


def get_text_chunks(text):
    """Split text into overlapping chunks so each answer keeps its context."""
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    return splitter.split_text(text)


def get_vectorstore(text_chunks):
    """Embed the chunks with OpenAI and store them in an in-memory Chroma index."""
    return Chroma.from_texts(text_chunks, OpenAIEmbeddings())


def get_qa_chain(vectorstore, k=TOP_K):
    """Build a RetrievalQA chain that answers from the k most similar chunks."""
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"), temperature=0)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)


def answer_question(chain, question):
    """Run one question through the chain and return the answer text."""
    return chain.invoke({"query": question})["result"]


def process_documents(pdf_docs):
    """Turn uploaded PDFs into a ready-to-use QA chain (sidebar 'Process' button)."""
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
        return
    if not pdf_docs:
        st.warning("Upload at least one PDF first.")
        return
    with st.spinner("Processing"):
        chunks = get_text_chunks(get_pdf_text(pdf_docs))
        if not chunks:
            st.error("No text found. Scanned PDFs need OCR before they can be read.")
            return
        st.session_state.qa_chain = get_qa_chain(get_vectorstore(chunks))
        st.session_state.chat_history = []
    st.success(f"Ready: {len(chunks)} chunks from {len(pdf_docs)} file(s).")


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.header("Chat with multiple PDFs :books:")

    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs, then click Process", type="pdf", accept_multiple_files=True
        )
        if st.button("Process"):
            process_documents(pdf_docs)

    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(message)

    question = st.chat_input("Ask a question about your documents")
    if not question:
        return
    if st.session_state.qa_chain is None:
        st.warning("Upload your PDFs and click Process first.")
        return

    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking"):
            answer = answer_question(st.session_state.qa_chain, question)
        st.write(answer)
    st.session_state.chat_history += [("user", question), ("assistant", answer)]


if __name__ == "__main__":
    main()
