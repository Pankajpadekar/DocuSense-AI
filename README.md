# 🤖 DocuSense AI — Multi-Source Agentic RAG Application

> Chat with your PDFs, Word Documents, and YouTube Videos — all in one place!

DocuSense AI is a **Agentic Retrieval-Augmented Generation (RAG)** application built with LangChain, Google Gemini, and FAISS. It lets you upload multiple document sources, merges them into a unified knowledge base, and answers your questions with source citations. If the answer isn't found in your documents, it **automatically falls back to a live web search** — making it truly agentic.

---

## 📸 Demo

<img width="1914" height="815" alt="Screenshot 2026-06-16 170737" src="https://github.com/user-attachments/assets/a751d2bb-11d2-4f1e-b4df-00543cb782a4" />

<img width="1902" height="906" alt="Screenshot 2026-06-16 170746" src="https://github.com/user-attachments/assets/3ba2e604-2c7b-4dab-a64e-cf138b4a0bd3" />


---

## ✨ Features

- 📄 **Multi-Source Ingestion** — Upload PDFs, Word (.docx) files, and YouTube video URLs simultaneously
- 🧠 **Unified Knowledge Base** — All sources are merged into a single FAISS vector store
- 🌐 **Agentic Web Search Fallback** — Automatically searches the web via DuckDuckGo if answer not found in documents
- 💬 **Conversation Memory** — Remembers past questions within a session (last 10 exchanges)
- 🗜️ **Contextual Compression** — Filters irrelevant chunks before answering for cleaner retrieval
- 📌 **Source Citations** — Every answer shows which document/source it came from
- 📊 **RAGAS Evaluation** — Built-in quality scoring (Faithfulness, Answer Relevancy, Context Precision)
- 🎙️ **Customizable Responses** — Choose Tone, Audience, and Response Format per query
- 🌍 **YouTube Auto-Translation** — Non-English transcripts are auto-detected and translated to English via Gemini

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| LLM | Google Gemini 2.5 Flash |
| Embeddings | Google Gemini Embedding 2 Preview |
| Vector Store | FAISS |
| Orchestration | LangChain (LCEL) |
| PDF Loader | PyPDFLoader |
| Word Loader | Docx2txtLoader |
| YouTube | youtube-transcript-api |
| Web Search | DuckDuckGo Search |
| RAG Evaluation | RAGAS |
| UI | Streamlit |

---

## ⚙️ How It Works

```
Source Upload (PDF / DOCX / YouTube URL)
        ↓
Text Extraction & Chunking (RecursiveCharacterTextSplitter)
        ↓
Embeddings (Google Gemini Embedding)
        ↓
FAISS Vector Store
        ↓
Retriever (Top-K similarity search)
        ↓
[Optional] Contextual Compression (LLMChainExtractor)
        ↓
Gemini 2.5 Flash LLM + Prompt (with memory & tone/format controls)
        ↓
Answer + Source Citations
        ↓
[If not found] → Agentic Web Search Fallback (DuckDuckGo) → Web Answer
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Google Gemini API Key (get it from [Google AI Studio](https://aistudio.google.com/))

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/docusense-ai.git
   cd docusense-ai
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate        # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**

   Create a `.env` file in the root directory:
   ```env
   GOOGLE_API_KEY=your_google_gemini_api_key_here
   ```

5. **Run the app**
   ```bash
   streamlit run chat_app.py
   ```

---

## 📦 Requirements

```txt
streamlit
langchain
langchain-google-genai
langchain-community
langchain-text-splitters
langchain-core
langchain-classic
faiss-cpu
youtube-transcript-api
python-docx
docx2txt
pypdf
ragas
datasets
duckduckgo-search
python-dotenv
```

---

## 🔧 Advanced Options Explained

| Option | Description |
|---|---|
| **Contextual Compression** | Uses LLM to extract only the relevant portion of each retrieved chunk |
| **Show Source Citations** | Appends a "Sources Used" section to every answer |
| **Agentic Web Search Fallback** | If RAG answer is not found, auto-triggers a DuckDuckGo web search |
| **Conversation Memory** | Maintains last 10 exchanges as context for follow-up questions |
| **RAGAS Evaluation** | Scores each answer on Faithfulness, Answer Relevancy, and Context Precision |

---

## 📊 RAGAS Metrics

When RAGAS Evaluation is enabled, each response is scored on:

- **Faithfulness** — Is the answer grounded in the retrieved context?
- **Answer Relevancy** — Does the answer actually address the question?
- **Context Precision** — Are the retrieved chunks relevant to the question?

Scores range from 0 to 1. Higher is better.

---

## 📁 Project Structure

```
docusense-ai/
├── chat_app.py          # Main Streamlit application
├── .env                 # API keys (not committed to git)
├── .gitignore
├── requirements.txt
├── screenshots/
│   ├── home.png
│   └── advanced_options.png
└── README.md
```

## 🙋‍♂️ Author

**Pankaj** — Software Engineer
LinkedIn - https://www.linkedin.com/in/pankaj-padekar/

---

## 📄 License

This project is licensed under the MIT License.
