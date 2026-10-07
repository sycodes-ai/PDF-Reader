# PDF-Reader: Chat with your PDFs

A small RAG (retrieval-augmented generation) app: upload one or more PDFs, then ask questions and get answers taken from those documents.

Built with **Streamlit**, **LangChain**, **OpenAI** and **Chroma**.

## How it works

1. **Upload** PDFs in the Streamlit sidebar
2. **Extract** the text of each page with PyPDF2
3. **Chunk** the text into 1,000-character pieces with 200 characters of overlap
4. **Embed** the chunks with OpenAI embeddings
5. **Store** the vectors in an in-memory Chroma index
6. **Answer** each question with a LangChain RetrievalQA chain that sends the most similar chunks to the chat model

## Run it locally

```bash
git clone https://github.com/sycodes-ai/PDF-Reader.git
cd PDF-Reader
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then put your OpenAI API key in .env
streamlit run pdf_reader.py
```

`OPENAI_MODEL` in `.env` is optional (default: `gpt-3.5-turbo`).

## Limitations and next steps

- Works with PDFs that have a text layer; scanned PDFs need OCR first.
- The index lives in memory, so documents must be processed again after a restart.
- Next: a test set of questions to measure answer quality, source citations, and failure tagging.
