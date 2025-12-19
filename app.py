import os
import json
import numpy as np
import streamlit as st
from PIL import Image
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# YOLO
# =========================
from ultralytics import YOLO

# =========================
# LangChain
# =========================
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# =========================
# OpenAI
# =========================
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found. Check .env file.")

client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Vision–RAG Agricultural Decision Support",
    layout="wide"
)

MODEL_PATH = "best.pt"
KNOWLEDGE_DIR = "data/knowledge"

# =========================
# LOAD MODELS (CACHED)
# =========================
@st.cache_resource
def load_yolo():
    return YOLO(MODEL_PATH)

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource
def build_vector_db():
    docs = []

    for file in os.listdir(KNOWLEDGE_DIR):
        path = os.path.join(KNOWLEDGE_DIR, file)

        if file.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for sec in data.get("sections", []):
                if sec.get("content", "").strip():
                    docs.append(
                        Document(
                            page_content=sec["content"],
                            metadata={"source": file}
                        )
                    )

        elif file.endswith(".txt"):
            docs.extend(TextLoader(path, encoding="utf-8").load())

        elif file.endswith(".pdf"):
            docs.extend(PyPDFLoader(path).load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)
    return FAISS.from_documents(chunks, embeddings)


yolo_model = load_yolo()
embeddings = load_embeddings()
vector_db = build_vector_db()

# =========================
# HELPER FUNCTIONS
# =========================
def parse_yolo_labels(counter):
    parsed = []
    for label, count in counter.items():
        parts = label.split("_")
        parsed.append({
            "fruit": "_".join(parts[1:]).lower(),
            "status": parts[0].lower(),
            "count": count
        })
    return parsed


def retrieve_context(query, k=3):
    docs = vector_db.similarity_search(query, k=k)
    return "\n".join(d.page_content for d in docs)


def build_initial_prompt(facts, context):
    return f"""
You are an agricultural quality-control assistant.

DETECTED FACTS:
{facts}

SOP CONTEXT:
{context}

TASK:
Explain the recommended operational actions clearly in English.
"""


def build_chat_prompt(question, facts, context):
    return f"""
You are an agricultural decision-support assistant.

KNOWN FACTS:
{facts}

SOP CONTEXT:
{context}

QUESTION:
{question}

TASK:
Answer clearly in English.
If the information is not specified in the SOP, explicitly state it.
"""

# =========================
# EVALUATION (INITIAL ONLY)
# =========================
def factual_consistency_score(decisions, llm_response):
    score, total = 0, 0
    text = llm_response.lower()
    for d in decisions:
        for k in [d["fruit"], d["status"], str(d["count"])]:
            total += 1
            if k in text:
                score += 1
    return score / total if total else 0


def sop_grounding_score(llm_response, sop_context):
    emb_r = embeddings.embed_query(llm_response)
    emb_c = embeddings.embed_query(sop_context)
    return cosine_similarity(
        np.array(emb_r).reshape(1, -1),
        np.array(emb_c).reshape(1, -1)
    )[0][0]


def actionability_score(text):
    keywords = [
        "temperature", "°c", "humidity", "ppm",
        "hours", "days", "store", "transport", "ripening"
    ]
    return sum(1 for k in keywords if k in text.lower()) / len(keywords)


def readability_score(text):
    s = [x for x in text.split(".") if x.strip()]
    avg = sum(len(x.split()) for x in s) / max(len(s), 1)
    if 15 <= avg <= 25:
        return 1.0
    elif 10 <= avg <= 30:
        return 0.7
    return 0.4


def final_score(fc, sg, act, read):
    return 0.35*fc + 0.30*sg + 0.20*act + 0.15*read

# =========================
# SESSION STATE INIT
# =========================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "initial_done" not in st.session_state:
    st.session_state.initial_done = False

if "boxed_image" not in st.session_state:
    st.session_state.boxed_image = None

if "facts" not in st.session_state:
    st.session_state.facts = None

if "decisions" not in st.session_state:
    st.session_state.decisions = None

if "label_summary" not in st.session_state:
    st.session_state.label_summary = None

# =========================
# SIDEBAR — KNOWLEDGE BASE
# =========================
st.sidebar.title("📚 Knowledge Base")

st.sidebar.subheader("Active Documents")
for f in os.listdir(KNOWLEDGE_DIR):
    col1, col2 = st.sidebar.columns([4,1])
    col1.write(f)
    if col2.button("❌", key=f):
        os.remove(os.path.join(KNOWLEDGE_DIR, f))
        st.cache_resource.clear()
        st.rerun()

uploaded = st.sidebar.file_uploader(
    "Add SOP document",
    type=["pdf","txt","json"]
)

if uploaded:
    st.sidebar.info("File selected. Confirm to add or ignore to cancel.")
    if st.sidebar.button("Confirm upload"):
        with open(os.path.join(KNOWLEDGE_DIR, uploaded.name), "wb") as f:
            f.write(uploaded.getbuffer())
        st.cache_resource.clear()
        st.rerun()

# =========================
# MAIN UI
# =========================
st.title("Vision–RAG Agricultural Chatbot")

image_file = st.file_uploader(
    "Upload fruit image",
    type=["jpg","png"],
    disabled=st.session_state.initial_done
)

# =========================
# INITIAL DETECTION (RUN ONCE)
# =========================
if image_file and not st.session_state.initial_done:
    img = Image.open(image_file)

    results = yolo_model.predict(img, conf=0.3)
    boxed = results[0].plot()
    boxed = Image.fromarray(boxed[..., ::-1])  # BGR → RGB

    st.session_state.boxed_image = boxed

    counter = Counter()
    for box in results[0].boxes:
        counter[yolo_model.names[int(box.cls[0])]] += 1

    parsed = parse_yolo_labels(counter)
    facts = "\n".join(f"{p['count']} {p['status']} {p['fruit']}" for p in parsed)
    st.session_state.label_summary = parsed

    context = retrieve_context(" ".join(counter.keys()))
    prompt = build_initial_prompt(facts, context)

    res = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    answer = res.choices[0].message.content

    fc = factual_consistency_score(parsed, answer)
    sg = sop_grounding_score(answer, context)
    act = actionability_score(answer)
    read = readability_score(answer)
    score = final_score(fc, sg, act, read)

    st.session_state.facts = facts
    st.session_state.decisions = parsed

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "evaluation": (fc, sg, act, read, score)
    })

    st.session_state.initial_done = True
    st.rerun()

# =========================
# SHOW BOUNDING BOX (ALWAYS)
# =========================
if st.session_state.boxed_image is not None:
    st.subheader("🖼 Detection Result")
    st.image(
        st.session_state.boxed_image,
        caption="Detected fruits with bounding boxes",
        width=400
    )
    if st.session_state.label_summary:
        st.markdown("### 📊 Detection Summary")

        for item in st.session_state.label_summary:
            st.write(
                f"- **{item['count']}×** {item['status'].capitalize()} "
                f"{item['fruit'].capitalize()}"
            )

# =========================
# CHAT UI
# =========================
st.divider()
st.subheader("💬 Chat")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "evaluation" in msg:
            fc, sg, act, read, score = msg["evaluation"]
            st.caption(
                f"Evaluation — FC:{fc:.2f} | SOP:{sg:.2f} | "
                f"ACT:{act:.2f} | READ:{read:.2f} | FINAL:{score:.2f}"
            )

question = st.chat_input("Ask a follow-up question")

if question and st.session_state.facts:
    st.session_state.chat_history.append({
        "role": "user",
        "content": question
    })

    ctx = retrieve_context(question)
    chat_prompt = build_chat_prompt(
        question,
        st.session_state.facts,
        ctx
    )

    chat_res = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": chat_prompt}],
        temperature=0.2
    )

    answer = chat_res.choices[0].message.content

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer
    })

    st.rerun()
