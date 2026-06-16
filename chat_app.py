import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from datasets import Dataset

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_community.tools import DuckDuckGoSearchRun
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_core.documents import Document
from ragas import evaluate
from ragas.metrics.collections import faithfulness, answer_relevancy, context_precision

load_dotenv()

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Source RAG Chat",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 DocuSense AI")
st.caption("Chat with PDF, YouTube Videos, and Word Documents — all in one place!")

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:

    st.header("📂 Add Your Sources")
    st.info("You can add one or more sources. All will be combined into a single knowledge base.")

    st.subheader("📄 PDF File")
    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

    st.subheader("📝 Word File (.docx)")
    uploaded_docx = st.file_uploader("Upload a Word document", type=["docx"])

    st.subheader("🎥 YouTube Video")
    yt_input = st.text_input("Paste YouTube URL or Video ID", placeholder="e.g. https://youtube.com/watch?v=abc123")

    st.divider()

    st.header("⚙️ Customize Response")
    tone = st.selectbox("🎙️ Tone", ["Professional", "Simple & Beginner-Friendly", "Concise", "Detailed & In-Depth"])
    audience = st.selectbox("👤 Audience", ["General", "Student / Fresher", "Software Engineer", "Senior Engineer"])
    format_style = st.selectbox("📋 Response Format", ["Paragraph", "Bullet Points", "Step-by-Step", "With Examples"])

    st.divider()

    st.header("🔧 Advanced Options")
    use_compression = st.toggle("Contextual Compression", value=False, help="Filters irrelevant content from retrieved chunks")
    show_sources = st.toggle("Show Source Citations", value=True, help="Show which source the answer came from")
    use_web_fallback = st.toggle("🌐 Agentic Web Search Fallback", value=True, help="Auto web search if answer not in documents")
    use_memory = st.toggle("🧠 Conversation Memory", value=True, help="Remember past questions in this session")
    show_ragas = st.toggle("📊 Show RAGAS Evaluation", value=False, help="Evaluate answer quality using RAGAS metrics (slower)")

    st.divider()
    if st.button("🗑️ Clear All & Start Over"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ─── Helper: Extract YouTube Video ID ────────────────────────────────────────
def extract_video_id(url_or_id: str) -> str:
    url_or_id = url_or_id.strip()
    if "youtube.com/watch?v=" in url_or_id:
        return url_or_id.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url_or_id:
        return url_or_id.split("youtu.be/")[1].split("?")[0]
    else:
        return url_or_id


# ─── Helper: Translate to English ────────────────────────────────────────────
def translate_to_english(text: str, llm) -> str:
    detect_prompt = f"Detect the language of the following text and if it is NOT English, translate it to English. If it is already in English, return it as-is.\n\nText:\n{text[:3000]}"
    result = llm.invoke(detect_prompt)
    return result.content


# ─── Helper: Load YouTube Transcript ─────────────────────────────────────────
def load_youtube_transcript(video_id: str, llm) -> list:
    ytt_api = YouTubeTranscriptApi()
    try:
        transcript_list = ytt_api.fetch(video_id, languages=["en"])
        transcript = " ".join([chunk.text for chunk in transcript_list])
    except Exception:
        transcript_list = ytt_api.fetch(video_id)
        raw_text = " ".join([chunk.text for chunk in transcript_list])
        transcript = translate_to_english(raw_text, llm)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])
    for chunk in chunks:
        chunk.metadata["source"] = f"YouTube: {video_id}"
    return chunks


# ─── Helper: Load PDF ─────────────────────────────────────────────────────────
def load_pdf(file_bytes, file_name) -> list:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    os.unlink(tmp_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    for chunk in chunks:
        chunk.metadata["source"] = f"PDF: {file_name}"
    return chunks


# ─── Helper: Load DOCX ───────────────────────────────────────────────────────
def load_docx(file_bytes, file_name) -> list:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    loader = Docx2txtLoader(tmp_path)
    documents = loader.load()
    os.unlink(tmp_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    for chunk in chunks:
        chunk.metadata["source"] = f"Word: {file_name}"
    return chunks


# ─── Build Vector Store ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔄 Building knowledge base from your sources...")
def build_vectorstore(pdf_name, pdf_bytes, docx_name, docx_bytes, yt_video_id):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2-preview")

    all_chunks = []

    if pdf_bytes:
        pdf_chunks = load_pdf(pdf_bytes, pdf_name)
        all_chunks.extend(pdf_chunks)
        st.sidebar.success(f"✅ PDF loaded: {len(pdf_chunks)} chunks")

    if docx_bytes:
        docx_chunks = load_docx(docx_bytes, docx_name)
        all_chunks.extend(docx_chunks)
        st.sidebar.success(f"✅ Word doc loaded: {len(docx_chunks)} chunks")

    if yt_video_id:
        try:
            yt_chunks = load_youtube_transcript(yt_video_id, llm)
            all_chunks.extend(yt_chunks)
            st.sidebar.success(f"✅ YouTube loaded: {len(yt_chunks)} chunks")
        except TranscriptsDisabled:
            st.sidebar.error("❌ Transcripts disabled for this video.")
        except Exception as e:
            st.sidebar.error(f"❌ YouTube error: {str(e)}")

    if not all_chunks:
        return None

    vectorstore = FAISS.from_documents(all_chunks, embeddings)
    return vectorstore


# ─── Build Dynamic Prompt WITH Memory ────────────────────────────────────────
def build_prompt_with_memory(tone, audience, format_style, show_sources):
    """Prompt that includes chat history for memory"""
    citation_instruction = "\nAfter your answer, add a 'Sources Used:' section listing which sources the information came from." if show_sources else ""

    system_template = f"""You are a helpful AI assistant helping a {{audience}}.
Tone: {{tone}}
Response Format: {{format_style}}

Use ONLY the context provided below to answer the question.
If the answer is not in the context, say "I couldn't find this in the provided sources."
{citation_instruction}

Context:
{{context}}"""

    return ChatPromptTemplate.from_messages([
        ("system", system_template),
        MessagesPlaceholder(variable_name="chat_history"),  # ← memory injected here
        ("human", "{question}")
    ])


# ─── Format Docs with Source Info ────────────────────────────────────────────
def format_docs(docs):
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        formatted.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n".join(formatted)


# ─── Web Search Fallback (Agentic RAG) ───────────────────────────────────────
def web_search_answer(question: str, llm) -> str:
    search = DuckDuckGoSearchRun()
    search_results = search.run(question)
    web_prompt = f"""You are a helpful assistant. Answer the question using the web search results below.
Always mention that this answer came from a web search since it was not found in the uploaded documents.

Web Search Results:
{search_results}

Question: {question}

Answer:"""
    result = llm.invoke(web_prompt)
    return f"🌐 **Answer from Web Search** (not found in documents):\n\n{result.content}"


def is_answer_found(answer: str) -> bool:
    not_found_phrases = [
        "i couldn't find",
        "not in the provided sources",
        "not found in",
        "i don't know",
        "no information",
        "cannot find",
        "not available in"
    ]
    return not any(phrase in answer.lower() for phrase in not_found_phrases)


# ─── RAGAS Evaluation ────────────────────────────────────────────────────────
def run_ragas_evaluation(question: str, answer: str, retrieved_docs: list):
    """Evaluate RAG quality using RAGAS metrics"""
    try:
        contexts = [doc.page_content for doc in retrieved_docs]
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }
        dataset = Dataset.from_dict(data)
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
        return result
    except Exception as e:
        return None


# ─── Check Sources ────────────────────────────────────────────────────────────
has_pdf = uploaded_pdf is not None
has_docx = uploaded_docx is not None
has_yt = bool(yt_input and yt_input.strip())

if not any([has_pdf, has_docx, has_yt]):
    st.info("👈 Please add at least one source from the sidebar to get started.")
    st.markdown("""
    ### You can add:
    - 📄 **PDF** — any PDF document
    - 📝 **Word file** — any .docx document
    - 🎥 **YouTube video** — paste the URL or video ID
    """)
    st.stop()

# ─── Build Knowledge Base ─────────────────────────────────────────────────────
video_id = extract_video_id(yt_input) if has_yt else None

vectorstore = build_vectorstore(
    pdf_name=uploaded_pdf.name if has_pdf else None,
    pdf_bytes=uploaded_pdf.read() if has_pdf else None,
    docx_name=uploaded_docx.name if has_docx else None,
    docx_bytes=uploaded_docx.read() if has_docx else None,
    yt_video_id=video_id
)

if vectorstore is None:
    st.error("❌ Could not build knowledge base. Please check your sources.")
    st.stop()

st.success("✅ Knowledge base ready! Start chatting below.")

active_sources = []
if has_pdf: active_sources.append(f"📄 {uploaded_pdf.name}")
if has_docx: active_sources.append(f"📝 {uploaded_docx.name}")
if has_yt: active_sources.append(f"🎥 YouTube: {video_id}")
st.caption(f"Active sources: {' | '.join(active_sources)}")

# ─── Initialize Memory (Chat History) ────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # stores LangChain message objects for memory

# ─── Display Chat History ─────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ─── Chat Input ───────────────────────────────────────────────────────────────
user_question = st.chat_input("Ask anything from your sources...")

if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching your sources..."):
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

            # Retriever
            base_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

            if use_compression:
                compressor = LLMChainExtractor.from_llm(llm)
                retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=base_retriever
                )
            else:
                retriever = base_retriever

            # Retrieve docs (needed for RAGAS)
            retrieved_docs = retriever.invoke(user_question)
            context_text = format_docs(retrieved_docs)

            # Build prompt with memory
            prompt = build_prompt_with_memory(tone, audience, format_style, show_sources)

            # Get chat history for memory
            chat_history = st.session_state.chat_history if use_memory else []

            # Build chain
            chain = prompt | llm | StrOutputParser()

            answer = chain.invoke({
                "context": context_text,
                "question": user_question,
                "chat_history": chat_history,
                "tone": tone,
                "audience": audience,
                "format_style": format_style
            })

        # ── Agentic Fallback ──
        if use_web_fallback and not is_answer_found(answer):
            with st.spinner("📄 Not found in documents. Searching the web..."):
                answer = web_search_answer(user_question, llm)

        st.markdown(answer)

        # ── Update Memory ──
        if use_memory:
            st.session_state.chat_history.append(HumanMessage(content=user_question))
            st.session_state.chat_history.append(AIMessage(content=answer))

            # Keep only last 10 exchanges (20 messages) to avoid context overflow
            if len(st.session_state.chat_history) > 20:
                st.session_state.chat_history = st.session_state.chat_history[-20:]

        st.session_state.messages.append({"role": "assistant", "content": answer})

        # ── RAGAS Evaluation ──
        if show_ragas and is_answer_found(answer):
            with st.spinner("📊 Running RAGAS evaluation..."):
                ragas_result = run_ragas_evaluation(user_question, answer, retrieved_docs)
                if ragas_result:
                    with st.expander("📊 RAGAS Evaluation Score"):
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Faithfulness", f"{ragas_result['faithfulness']:.2f}")
                        col2.metric("Answer Relevancy", f"{ragas_result['answer_relevancy']:.2f}")
                        col3.metric("Context Precision", f"{ragas_result['context_precision']:.2f}")
                        st.caption("Scores range from 0 to 1. Higher is better.")