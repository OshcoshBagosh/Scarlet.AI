#!/usr/bin/env python3
"""
Rutgers AI Assistant - Prompto (Integrated Enhanced UI + Advanced Features)
Combines polished UI design with advanced RAG capabilities:
- Confidence scoring & hallucination detection
- Claim-level evidence checking
- Policy-aware/risk-aware responses
- Topic classification for safety modulation
"""

import os
import time
import traceback
import random
import base64
from typing import List, Dict

import requests
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions

# --- Docling imports ---
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from docling.datamodel.document import InputDocument

# --- Ollama (LLM + optional embeddings) ---
import ollama

# --- Explainability & Evaluation modules (ADVANCED FEATURES) ---
from explainability_module import (
    calculate_confidence_score,
    detect_hallucination,
    generate_confidence_badge_html,
    generate_hallucination_warning_html,
    format_sources_with_confidence,
    perform_claim_level_checking
)

from evaluation_module import (
    EvaluationLogger,
    render_evaluation_dashboard
)

# --- Topic Classification & Policy-Aware Response (ADVANCED FEATURES) ---
from topic_classifier import TopicClassifier
from policy_aware_module import (
    PolicyAwareResponder,
    create_risk_aware_rag_prompt,
    apply_policy_aware_modulation
)

def get_base64_image(image_path):
    """Load logo image for header"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

# =========================
# Config (env overrides)
# =========================
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
USE_OLLAMA_EMBED = os.getenv("USE_OLLAMA_EMBED", "0") == "1"
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))
NUM_CTX = int(os.getenv("NUM_CTX", "8192"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
TOP_K = int(os.getenv("TOP_K", "40"))
TOP_P = float(os.getenv("TOP_P", "0.9"))
N_RESULTS = int(os.getenv("N_RESULTS", "8"))
MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "4000"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "25"))

# =========================
# Enhanced Quick Questions with Answers
# =========================
QUICK_QUESTIONS = {
    "What is in-state tuition?": {
        "answer": "For the 2024-2025 academic year, in-state tuition at Rutgers is approximately $16,000-$17,000 per year for full-time undergraduate students. This varies by school and program. Check the official tuition page for current rates.",
        "sources": [
            {"title": "Tuition and Fees", "url": "https://admissions.rutgers.edu/costs-and-aid/tuition-fees", "distance": 0.15},
            {"title": "Student Accounting", "url": "https://finance.rutgers.edu/student-abc", "distance": 0.18}
        ]
    },
    "How to apply for financial aid?": {
        "answer": "To apply for financial aid at Rutgers:\n1. Complete the FAFSA (Free Application for Federal Student Aid)\n2. Use Rutgers school code 002629\n3. Complete additional required forms through the Rutgers financial aid portal\n4. Submit documents by the priority deadline for maximum aid consideration",
        "sources": [
            {"title": "Financial Aid Application", "url": "https://scarlethub.rutgers.edu/financial-services/apply-for-aid/how-to-apply/", "distance": 0.12},
            {"title": "FAFSA Information", "url": "https://scarlethub.rutgers.edu/financial-services/", "distance": 0.16}
        ]
    },
    "Where is the student center?": {
        "answer": "Rutgers has multiple student centers:\n• **College Avenue**: Rutgers Student Center\n• **Livingston**: Livingston Student Center\n• **Busch**: Busch Student Center\n• **Cook/Douglas**: Cook Student Center\nEach offers dining, study spaces, meeting rooms, and student organization offices.",
        "sources": [
            {"title": "Student Centers", "url": "https://sca.rutgers.edu/student-centers", "distance": 0.10},
            {"title": "Campus Locations", "url": "https://newbrunswick.rutgers.edu/student-housing-and-dining", "distance": 0.14}
        ]
    },
    "What are dining hall hours?": {
        "answer": "Dining hall hours vary by location and semester. Generally:\n• **Weekdays**: 7:00 AM - 8:00 PM\n• **Weekends**: 9:00 AM - 7:00 PM\n• Some locations have extended hours\nCheck the Rutgers Food Services website for current hours.",
        "sources": [
            {"title": "Dining Locations & Hours", "url": "https://food.rutgers.edu/places-eat", "distance": 0.11},
            {"title": "Meal Plan Information", "url": "https://food.rutgers.edu/meal-plans", "distance": 0.15}
        ]
    },
    "How to contact academic advising?": {
        "answer": "Contact academic advising through:\n1. Your specific school's advising office (SAS, SOE, RBS, etc.)\n2. Schedule appointments through Starfish or your school's portal\n3. Visit advising offices during walk-in hours\n4. Email your assigned advisor directly",
        "sources": [
            {"title": "SAS Advising", "url": "https://sas.rutgers.edu/about/sas-offices/detail/office-of-advising-and-academic-services", "distance": 0.13},
            {"title": "Academic Support", "url": "https://sasundergrad.rutgers.edu/", "distance": 0.17}
        ]
    },
    "When is the add/drop period?": {
        "answer": "The add/drop period is typically the first 10 days of each semester. During this period, you can add or drop courses without penalty. Check the academic calendar for exact dates each semester.",
        "sources": [
            {"title": "Academic Calendar", "url": "https://scheduling.rutgers.edu/academic-calendar/", "distance": 0.09},
            {"title": "Registration Policies", "url": "https://nbregistrar.rutgers.edu/", "distance": 0.12}
        ]
    },
    "Where is the health center?": {
        "answer": "Rutgers Health Services has multiple locations:\n• **Hurtado Health Center** on College Avenue\n• **Busch-Livingston Health Center**\n• **Cook-Douglass Health Center**\nAll centers provide medical care, counseling, immunizations, and wellness services.",
        "sources": [
            {"title": "Health Services Locations", "url": "https://health.rutgers.edu/about-us/hours-and-locations", "distance": 0.08},
            {"title": "Medical Services", "url": "https://health.rutgers.edu/medical-and-counseling-services/medical-services", "distance": 0.11}
        ]
    },
    "How to join a student club?": {
        "answer": "To join a student club:\n1. Browse organizations on GetInvolved.Rutgers.edu\n2. Attend the Student Involvement Fair each semester\n3. Contact club leaders through social media or email\n4. Attend meetings and events\nThere are over 500 student organizations!",
        "sources": [
            {"title": "Student Organizations", "url": "https://sca.rutgers.edu/campus-involvement/student-organizations", "distance": 0.10},
            {"title": "Get Involved Portal", "url": "https://involvement.rutgers.edu/", "distance": 0.13}
        ]
    }
}

CAMPUS_TRIVIA = [
    "Did you know? Rutgers was founded in 1766!",
    "Fun fact: The Scarlet Knight became mascot in 1955",
    "Rutgers is the 8th oldest college in the United States",
    "Rutgers has three campuses: New Brunswick, Newark, and Camden",
    "The first intercollegiate football game was at Rutgers in 1869",
]

DEPARTMENTS = {
    "Registrar": "https://nbregistrar.rutgers.edu/",
    "Financial Aid": "https://scarlethub.rutgers.edu/financial-services/",
    "Housing": "https://ruoncampus.rutgers.edu/",
    "Health Services": "https://health.rutgers.edu/",
    "Career Services": "https://careers.rutgers.edu/",
    "Student Affairs": "https://studentaffairs.rutgers.edu/",
    "Academic Advising": "https://sas.rutgers.edu/academics"
}

IMPORTANT_DATES = {
    "Add/Drop Period": "Sep 2-11",
    "Thanksgiving Break": "Nov 27-30",
    "Winter Break": "Dec 23 - Jan 21",
    "Spring Registration": "Nov 1",
    "Finals": "Dec 15-22",
    "Commencement": "May 17"
}

# =========================
# Rutgers URLs (full list maintained)
# =========================
RUTGERS_URLS = [
    "https://classes.rutgers.edu/soc/#school?code=01&semester=92025&campus=NB&level=U",
    "https://nbregistrar.rutgers.edu/",
    "https://scheduling.rutgers.edu/academic-calendar/",
    "https://scheduling.rutgers.edu/course-scheduling/standard-course-periods/",
    "https://scheduling.rutgers.edu/exam-scheduling/final-exams/",
    "https://scheduling.rutgers.edu/exam-scheduling/",
    "https://admissions.rutgers.edu/apply/dates-deadlines/new-brunswick",
    "https://commencement.rutgers.edu/",
    "https://sasundergrad.rutgers.edu/resources/policies/details/rutgers-university-guidelines/graduation-honors",
    "https://scarlethub.rutgers.edu/registrar/graduation-and-diploma-information/",
    "https://scarlethub.rutgers.edu/registrar/graduation-and-diploma-information/electronic-diplomas/",
    "https://scarlethub.rutgers.edu/registrar/academics-and-records/instant-enrollment-verification/",
    "https://finance.rutgers.edu/help/forms-and-templates",
    "https://policies.rutgers.edu/PublicPageViewText.aspx?id=148",
    "https://sis.rutgers.edu/tags/",
    "https://catalogs.rutgers.edu/generated/cam-grad_0709/pg102.html",
    "https://sasundergrad.rutgers.edu/majors-and-core-curriculum/credits/transfer-credits",
    "https://scarlethub.rutgers.edu/registrar/registration/withdrawal-from-all-courses/",
    "https://finance.rutgers.edu/student-abc/refunds/withdrawals-school",
    "https://www.ugadmissions.rutgers.edu/reenrollment/",
    "https://catalogs.rutgers.edu/generated/nb-ug_1315/pg679.html",
    "https://scarlethub.rutgers.edu/registrar/faculty-staff/",
    "https://finance.rutgers.edu/student-abc/refunds",
    "https://scarlethub.rutgers.edu/financial-services/forms-documents/",
    "https://scarlethub.rutgers.edu/financial-services/financial-aid-disbursement/",
    "https://finance.rutgers.edu/student-abc",
    "https://finservices.rutgers.edu/otb/",
    "https://scarlethub.rutgers.edu/financial-services/tools-resources/frequently-asked-questions/",
    "https://scarlethub.rutgers.edu/financial-services/tools-resources/financial-aid-student-portal-library/",
    "https://scarlethub.rutgers.edu/financial-services/office-of-financial-aid/",
    "https://scarlethub.rutgers.edu/financial-services/",
    "https://finance.rutgers.edu/student-abc/refunds/setting-direct-deposit",
    "https://scarlethub.rutgers.edu/forms/",
    "https://scarlethub.rutgers.edu/financial-services/student-abc/",
    "https://financialaid.rutgers.edu/financial-services/eligibility/enrollment-requirements/withdrawing-from-all-courses/",
    "https://scarlethub.rutgers.edu/financial-services/tools-resources/",
    "https://admissions.rutgers.edu/costs-and-aid/tuition-fees",
    "https://scarlethub.rutgers.edu/financial-services/apply-for-aid/how-to-apply/financial-aid-for-summer-courses/",
    "https://finance.rutgers.edu/student-abc/insurance-students/student-health-insurance-plan-ship",
    "https://scarlethub.rutgers.edu/financial-services/student-employment/students/federal-work-study-program-fwsp/",
    "https://admissions.rutgers.edu/costs-and-aid/scholarships",
    "https://newbrunswick.rutgers.edu/student-housing-and-dining",
    "https://ruoncampus.rutgers.edu/housing-information/continuing-student-housing",
    "https://ruoncampus.rutgers.edu/",
    "https://www.rutgers.edu/housing-and-dining",
    "https://food.rutgers.edu/places-eat",
    "https://food.rutgers.edu/meal-plans",
    "https://food.rutgers.edu/meal-plan-faq",
    "https://food.rutgers.edu/ru-express",
    "https://food.rutgers.edu/places-eat/retail-dining-menus",
    "https://food.rutgers.edu/places-eat/dining-hall-tour",
    "https://sca.rutgers.edu/student-centers/places-eat",
    "https://food.rutgers.edu/",
    "https://ruoncampus.rutgers.edu/housing-info",
    "https://ruoncampus.rutgers.edu/housing-info/rates-and-billing",
    "https://ruoncampus.rutgers.edu/housing-info/new-student-housing-information",
    "https://ruoncampus.rutgers.edu/housing-info/spring-move-in-2025",
    "https://ruoncampus.rutgers.edu/housing-information/continuing-student-housing",
    "https://ruoncampus.rutgers.edu/housing-info/graduate-student-housing",
    "https://ruoncampus.rutgers.edu/housing-info/family-housing",
    "https://ruoncampus.rutgers.edu/housing-information/housing-cancellation",
    "https://ruoncampus.rutgers.edu/housing-info/special-accommodations",
    "https://ruoncampus.rutgers.edu/housing-info/end-year-hall-closing",
    "https://studentconduct.rutgers.edu/",
    "https://studentconduct.rutgers.edu/processes/university-code-student-conduct",
    "https://policies.rutgers.edu/B.aspx?BookId=11912",
    "https://studentconduct.rutgers.edu/processes",
    "https://studentconduct.rutgers.edu/about-us/contact-us",
    "https://studentconduct.rutgers.edu/faqs",
    "https://studentconduct.rutgers.edu/report-concern",
    "https://studentconduct.rutgers.edu/records-transcripts",
    "https://sca.rutgers.edu/campus-involvement/student-organizations",
    "https://sca.rutgers.edu/campus-involvement/get-involved-rutgers/getinvolvedrutgersedu",
    "https://newbrunswick.rutgers.edu/student-activities",
    "https://sca.rutgers.edu/campus-involvement",
    "https://studentaffairs.rutgers.edu/",
    "https://studentaffairs.rutgers.edu/resources",
    "https://involvement.rutgers.edu/",
    "https://sas.rutgers.edu/",
    "https://sas.rutgers.edu/academics",
    "https://sas.rutgers.edu/academics/majors-minors",
    "https://sasundergrad.rutgers.edu/",
    "https://sebs.rutgers.edu/",
    "https://sebs.rutgers.edu/academics",
    "https://soe.rutgers.edu/",
    "https://soe.rutgers.edu/academics",
    "https://www.masongross.rutgers.edu/",
    "https://www.business.rutgers.edu/",
    "https://www.business.rutgers.edu/undergraduate-new-brunswick",
    "https://bloustein.rutgers.edu/",
    "https://honorscollege.rutgers.edu/",
    "https://honorscollege.rutgers.edu/academics/overview",
    "https://health.rutgers.edu/",
    "https://health.rutgers.edu/immunizations",
    "https://caps.rutgers.edu/",
    "https://health.rutgers.edu/medical-and-counseling-services/medical-services",
    "https://health.rutgers.edu/medical-and-counseling-services/counseling-services",
    "https://health.rutgers.edu/about-us/hours-and-locations",
    "https://titleix.rutgers.edu/",
    "https://titleix.rutgers.edu/report",
    "https://titleix.rutgers.edu/resources",
    "https://rupd.rutgers.edu/",
    "https://ipo.rutgers.edu/transportation",
    "https://ipo.rutgers.edu/transportation/buses/nb",
]

# [EMBEDDING, LLM, DATA LOADING, VECTOR DB functions remain identical - keeping your existing implementation]
class OllamaEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self, model: str = OLLAMA_EMBED_MODEL):
        self.model = model
    def __call__(self, texts: List[str]):
        embs = []
        for t in texts:
            r = ollama.embeddings(model=self.model, prompt=t)
            embs.append(r["embedding"])
        return embs

def get_embedding_function():
    if USE_OLLAMA_EMBED:
        return OllamaEmbeddingFunction(OLLAMA_EMBED_MODEL)
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-small-en-v1.5"
    )

def call_llm(system_prompt: str, user_prompt: str, temperature: float = TEMPERATURE) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    res = ollama.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        options={
            "temperature": float(temperature),
            "top_k": TOP_K,
            "top_p": TOP_P,
            "num_predict": int(MAX_NEW_TOKENS),
            "num_ctx": int(NUM_CTX),
        },
    )
    return res["message"]["content"].strip()

def call_llm_stream(system_prompt: str, user_prompt: str, temperature: float = TEMPERATURE):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    stream = ollama.chat(
        model=OLLAMA_MODEL,
        messages=messages,
        stream=True,
        options={
            "temperature": float(temperature),
            "top_k": TOP_K,
            "top_p": TOP_P,
            "num_predict": int(MAX_NEW_TOKENS),
            "num_ctx": int(NUM_CTX),
        },
    )
    for chunk in stream:
        yield chunk["message"]["content"]

def fetch_ok(url: str, timeout: int = REQUEST_TIMEOUT) -> bool:
    try:
        r = requests.head(url, timeout=timeout, allow_redirects=True)
        return r.ok
    except Exception:
        return False

def convert_urls_resilient(urls: List[str]) -> List[tuple]:
    docs_with_urls: List[tuple] = []
    conv = DocumentConverter()
    for i, url in enumerate(urls):
        try:
            if not fetch_ok(url):
                st.warning(f"Skipping (HEAD failed): {url}")
                continue
            res = conv.convert(url, raises_on_error=False)
            if res and res.document:
                docs_with_urls.append((res.document, url))
            else:
                st.warning(f"Docling conversion yielded no document: {url}")
        except Exception as e:
            st.warning(f"Docling convert failed for {url}: {e}")
    return docs_with_urls

def chunk_documents(docs_with_urls: List[tuple]) -> List:
    chunker = HybridChunker(
        max_tokens=8191,
        merge_peers=True,
    )
    all_chunks = []
    for doc, original_url in docs_with_urls:
        try:
            for ch in chunker.chunk(dl_doc=doc):
                if hasattr(ch, "text"):
                    ch.text = ch.text[:MAX_CHUNK_CHARS]
                all_chunks.append((ch, original_url))
        except Exception as e:
            st.warning(f"Chunking failed: {e}")
    return all_chunks

def build_metadata(chunk, url_fallback: str):
    title = "Rutgers Document"
    filename = "doc"
    try:
        if hasattr(chunk, "meta"):
            m = chunk.meta
            if hasattr(m, "headings") and m.headings:
                title = m.headings[0]
            elif hasattr(m, "title") and m.title:
                title = m.title
            if hasattr(m, "origin") and hasattr(m.origin, "filename") and m.origin.filename:
                filename = m.origin.filename
    except Exception:
        pass
    if title == "Rutgers Document":
        title = filename.replace("-", " ").replace("_", " ").title()
    return {"title": title, "url": url_fallback}

@st.cache_resource
def setup_chromadb(force_rebuild: bool = False):
    client = chromadb.PersistentClient(path="./chroma_db")
    emb_fn = get_embedding_function()
    if not force_rebuild:
        try:
            col = client.get_collection("rutgers_docs", embedding_function=emb_fn)
            if col.count() > 0:
                return col
        except Exception:
            pass
    try:
        client.delete_collection("rutgers_docs")
    except Exception:
        pass
    col = client.create_collection(
        name="rutgers_docs",
        metadata={"description": "Rutgers University documents and information"},
        embedding_function=emb_fn,
    )
    with st.spinner("Converting Rutgers URLs with Docling…"):
        docs_with_urls = convert_urls_resilient(RUTGERS_URLS)
    with st.spinner("Chunking documents…"):
        chunks = chunk_documents(docs_with_urls)
    documents, ids, metadatas = [], [], []
    for i, item in enumerate(chunks):
        try:
            ch, original_url = item
        except Exception:
            ch = item
            original_url = ""
        txt = getattr(ch, "text", "")
        if not txt.strip():
            continue
        url = original_url
        try:
            if hasattr(ch, "meta") and hasattr(ch.meta, "origin"):
                if hasattr(ch.meta.origin, "source") and ch.meta.origin.source:
                    origin_src = ch.meta.origin.source
                    url = origin_src if isinstance(origin_src, str) else str(origin_src)
                elif hasattr(ch.meta.origin, "uri") and ch.meta.origin.uri:
                    url = str(ch.meta.origin.uri)
                elif hasattr(ch.meta.origin, "url") and ch.meta.origin.url:
                    url = str(ch.meta.origin.url)
        except Exception:
            pass
        meta = build_metadata(ch, url)
        documents.append(txt)
        ids.append(f"rutgers_chunk_{i+1}")
        metadatas.append(meta)
    if documents:
        with st.spinner(f"Adding {len(documents)} chunks to Chroma…"):
            col.add(documents=documents, ids=ids, metadatas=metadatas)
        st.success(f"✅ Loaded {len(documents)} chunks into ChromaDB")
    else:
        st.warning("No documents were added to ChromaDB.")
    return col

def search_semantic(col, query: str, n_results: int = N_RESULTS):
    res = col.query(query_texts=[query], n_results=n_results)
    out = []
    if not res or not res.get("documents"):
        return out
    docs = res["documents"][0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    for i in range(len(docs)):
        out.append({
            "text": docs[i],
            "title": metas[i].get("title", "Source"),
            "url": metas[i].get("url", ""),
            "distance": dists[i] if i < len(dists) else None,
        })
    return out

# =========================
# Enhanced UI Functions
# =========================
def typing_animation():
    """Show typing animation while processing"""
    placeholder = st.empty()
    for i in range(3):
        placeholder.markdown("🤔 Searching Rutgers knowledge base" + "." * (i + 1))
        time.sleep(0.5)
    placeholder.empty()
    return placeholder

def format_smart_response(answer, sources):
    """Detect response type and format accordingly"""
    urgent_keywords = ['emergency', 'deadline', 'urgent', 'crisis', 'immediately']
    if any(keyword in answer.lower() for keyword in urgent_keywords):
        st.markdown('<div style="background:#fff3cd;border:2px solid #ffc107;padding:12px;border-radius:12px;margin:10px 0;color:#856404;font-weight:bold;">🚨 ' + answer + '</div>', unsafe_allow_html=True)
    elif 'step' in answer.lower() or ('first' in answer.lower() and 'then' in answer.lower()):
        steps = answer.split('\n')
        st.markdown("### 📋 Step-by-Step Guide:")
        for step in steps:
            if step.strip():
                st.markdown(f"• {step.strip()}")
    else:
        st.markdown(answer)

def render_quick_questions():
    """Render quick action buttons"""
    st.markdown("### 🚀 Quick Questions")
    cols = st.columns(2)
    questions = list(QUICK_QUESTIONS.keys())
    for i, question in enumerate(questions):
        with cols[i % 2]:
            if st.button(f"• {question}", key=f"quick_{i}", use_container_width=True, type="secondary"):
                st.session_state.messages.append({"role": "user", "content": question})
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": QUICK_QUESTIONS[question]["answer"],
                    "sources": QUICK_QUESTIONS[question]["sources"],
                    "confidence": "High",
                    "is_quick_answer": True
                })
                st.rerun()

def render_department_links():
    """Render department quick links"""
    st.sidebar.markdown("### 📞 Quick Departments")
    for dept, url in DEPARTMENTS.items():
        st.sidebar.markdown(f'<a href="{url}" target="_blank" style="color:#FFDDDD;text-decoration:none;">📎 {dept}</a>', unsafe_allow_html=True)

def render_academic_timeline():
    """Render academic timeline"""
    st.sidebar.markdown("### 📅 Academic Timeline")
    for event, date in IMPORTANT_DATES.items():
        st.sidebar.write(f"**{event}:** {date}")

def render_emergency_contacts():
    """Render emergency contacts"""
    if st.sidebar.button("🚨 Emergency Contacts"):
        st.sidebar.markdown("""
        **Campus Police:** (848) 932-7211  
        **Health Services:** (848) 932-7402  
        **Counseling:** (848) 932-7884  
        **Title IX:** (848) 932-8200  
        **Any Emergency:** 911
        """)

def render_conversation_context():
    """Show recent conversation topics"""
    if len(st.session_state.messages) > 4:
        st.sidebar.markdown("### 💭 Conversation Context")
        st.sidebar.caption("Recent topics:")
        recent_topics = [msg["content"][:30] + "..." for msg in st.session_state.messages[-3:] if msg["role"] == "user"]
        for topic in recent_topics:
            st.sidebar.write(f"• {topic}")

def show_campus_trivia():
    """Show campus trivia every 2 interactions"""
    if "interaction_count" not in st.session_state:
        st.session_state.interaction_count = 0
    if "last_message_count" not in st.session_state:
        st.session_state.last_message_count = 0
    total_messages = len([msg for msg in st.session_state.messages if msg["content"] != "Welcome message shown"])
    if total_messages > st.session_state.last_message_count:
        if (len(st.session_state.messages) >= 2 and 
            st.session_state.messages[-1]["role"] == "assistant" and 
            st.session_state.messages[-2]["role"] == "user"):
            st.session_state.interaction_count += 1
            st.session_state.last_message_count = total_messages
            if st.session_state.interaction_count % 2 == 0:
                st.sidebar.markdown("---")
                st.sidebar.markdown("### 🎓 Campus Trivia")
                st.sidebar.markdown(f"*{random.choice(CAMPUS_TRIVIA)}*")

# =========================
# ADVANCED RAG ask function (WITH POLICY-AWARE + CLAIM CHECKING)
# =========================
def ask_rutgers_question(col, question: str, stream: bool = True, 
                        enable_claim_checking: bool = False, 
                        enable_policy_aware: bool = True):
    """
    ADVANCED RAG with confidence scoring, claim-checking, policy-aware responses
    """
    topic_classification = None
    policy_modulation = None
    
    if enable_policy_aware:
        if 'topic_classifier' not in st.session_state:
            st.session_state.topic_classifier = TopicClassifier(llm_client=ollama, model=OLLAMA_MODEL)
        topic_classification = st.session_state.topic_classifier.classify(question, use_llm=False)
    
    chunks = search_semantic(col, question, n_results=N_RESULTS)
    
    # ADVANCED: Confidence scoring
    confidence_level, emoji, color, avg_distance, calibrated_prob, calibration_info = calculate_confidence_score(
        chunks, use_calibration=False
    )
    
    confidence_data = {
        'level': confidence_level,
        'emoji': emoji,
        'color': color,
        'avg_distance': avg_distance,
        'calibrated_prob': calibrated_prob,
        'calibration_info': calibration_info
    }
    
    if not chunks:
        return "Sorry, I couldn't find relevant information about that topic in the Rutgers data.", [], None, confidence_data, True, None, None

    parts = []
    for ch in chunks:
        snippet = ch["text"]
        parts.append(f"[{ch['title']}]({ch['url']})\n{snippet}")

    context = "\n\n---\n\n".join(parts[:N_RESULTS])

    # ADVANCED: Policy-aware system prompt
    if enable_policy_aware and topic_classification:
        guidelines = st.session_state.topic_classifier.get_risk_guidelines(
            topic_classification['category_enum']
        )
        system_modifier = PolicyAwareResponder(st.session_state.topic_classifier).generate_system_prompt_modifier(
            topic_classification
        )
        system = (
            "You are a helpful Rutgers University assistant. "
            "Answer ONLY from the provided context. If the answer isn't there, say "
            "\"Sorry, I couldn't find relevant information about that topic in the Rutgers data.\" "
            "Include concise inline citations like [Title](URL) when applicable."
            + system_modifier
        )
    else:
        system = (
            "You are a helpful Rutgers University assistant. "
            "Answer ONLY from the provided context. If the answer isn't there, say "
            "\"Sorry, I couldn't find relevant information about that topic in the Rutgers data.\" "
            "Include concise inline citations like [Title](URL) when applicable."
        )

    user = (
        f"Question: {question}\n\n"
        f"Context (sources with snippets):\n{context}\n\n"
        "Answer:"
    )

    if stream:
        return None, chunks, call_llm_stream(system, user), confidence_data, False, None, topic_classification
    else:
        answer = call_llm(system, user)
        is_hallucination, reason = detect_hallucination(chunks, answer, confidence_level)
        claim_audit = None
        if enable_claim_checking:
            claim_verifications, claim_audit_html = perform_claim_level_checking(
                answer, col, llm_client=ollama, model=OLLAMA_MODEL
            )
            claim_audit = {
                'verifications': claim_verifications,
                'html': claim_audit_html
            }
        if enable_policy_aware and topic_classification:
            policy_modulation = apply_policy_aware_modulation(
                question, answer, chunks, st.session_state.topic_classifier
            )
        return answer, chunks, None, confidence_data, is_hallucination, claim_audit, policy_modulation

# =========================
# ENHANCED UI WITH ADVANCED FEATURES
# =========================
st.set_page_config(page_title="Prompto - Rutgers AI Assistant", page_icon="🏛️", layout="wide")

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #8B0000 0%, #A52A2A 25%, #B22222, #CD5C5C 75%, #DC143C 100%);
    background-size: 400% 400%; 
    animation: gradientShift 15s ease infinite;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    min-height: 100vh;
}

.header-container {
    position: relative;
    text-align: center;
    padding: 30px 0 30px 0;
    background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
    border-radius: 0 0 20px 20px;
    box-shadow: 0 4px 20px rgba(204, 0, 0, 0.1);
    margin-bottom: 20px;
    margin-top: 50px; 
    min-height: 200px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}

.logo-container {
    position: relative;
    z-index: 2;
    margin-bottom: -20px;
}

.title-container {
    position: relative;
    z-index: 1;
    margin-top: -10px;
    padding-top: 40px;
}

.main-title {
    font-size: 3.5em;
    font-weight: 900;
    background: linear-gradient(135deg, #CC0000, #8B0000);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 5px 0;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}

.main-subtitle {
    font-size: 1.3em;
    color: #666;
    margin-bottom: 15px;
    font-weight: 300;
}

.scarlet-divider {
    height: 4px;
    background: linear-gradient(90deg, #CC0000, #8B0000, #CC0000);
    margin: 10px auto 20px auto;
    width: 80%;
    border-radius: 2px;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #CC0000 0%, #8B0000 100%);
    color: white;
    padding-top: 2rem;
}

[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] strong {
    color: white !important;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] a {
    color: #FFDDDD !important;
}

[data-testid="stSidebar"] div.stButton > button {
    background: linear-gradient(135deg, #A00000, #8B0000);
    color: white;
    border: none;
    font-weight: bold;
    border-radius: 8px;
    padding: 10px 20px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(160, 0, 0, 0.3);
}

[data-testid="stSidebar"] div.stButton > button:hover {
    background: linear-gradient(135deg, #FF3333, #CC0000);
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(255, 51, 51, 0.4);
}

[data-testid="stTextInput"] > div > div > input {
    border: 2px solid #CC0000;
    border-radius: 12px;
    padding: 12px 18px;
    font-size: 1.1em;
    background: white;
    box-shadow: 0 4px 15px rgba(204, 0, 0, 0.1);
    transition: all 0.3s ease;
}

[data-testid="stTextInput"] > div > div > input:focus {
    border-color: #FF3333;
    box-shadow: 0 4px 20px rgba(204, 0, 0, 0.3);
    transform: translateY(-1px);
}

[data-testid="stChatMessage"] {
    background: white;
    color: #000000 !important;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    margin-bottom: 1.2rem;
    padding: 1rem 1.5rem;
    border: 1px solid #e0e0e0;
    transition: all 0.3s ease;
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] div,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] strong,
[data-testid="stChatMessage"] em {
    color: #000000 !important;
}

[data-testid="stChatMessage"]:hover {
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
    transform: translateY(-2px);
}

.source-box { 
    background: linear-gradient(135deg, #FFF5F5, #FFE8E8);
    padding: 1rem; 
    border-radius: 10px; 
    margin: .6rem 0; 
    border-left: 6px solid #CC0000;
    color: #000000; 
    box-shadow: 0 2px 8px rgba(204,0,0,0.1);
    transition: all 0.3s ease;
}

.source-box p,
.source-box div,
.source-box span {
    color: #000000 !important;
}

.source-box:hover {
    transform: translateX(5px);
    box-shadow: 0 4px 12px rgba(204,0,0,0.2);
}

.source-box a { 
    color: #990000 !important;
    text-decoration: none; 
    font-weight: bold; 
}

.header-logo {
    width: 100px;
    border-radius: 50%;
    box-shadow: 0 8px 25px rgba(204,0,0,0.3);
    border: 4px solid white;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

@keyframes float {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-5px); }
    100% { transform: translateY(0px); }
}

.logo-container img {
    animation: float 3s ease-in-out infinite;
}
</style>
""", unsafe_allow_html=True)

# Enhanced Header with Logo
try:
    img_base64 = get_base64_image("scarlet_logo.png")
    if img_base64:
        st.markdown(f"""
        <div class="header-container">
            <div class="logo-container">
                <img src='data:image/png;base64,{img_base64}' class='header-logo' alt='Prompto Logo'>
            </div>
            <div class="title-container">
                <h1 class="main-title">Prompto</h1>
                <p class="main-subtitle">Your AI Assistant for Rutgers University</p>
            </div>
            <div class="scarlet-divider"></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        raise FileNotFoundError
except:
    st.markdown("""
    <div class="header-container">
        <div class="title-container">
            <h1 class="main-title">Prompto</h1>
            <p class="main-subtitle">Your AI Assistant for Rutgers University</p>
        </div>
        <div class="scarlet-divider"></div>
    </div>
    """, unsafe_allow_html=True)

# Ollama health check
try:
    _ = ollama.list()
except Exception:
    st.error("Could not reach Ollama. Start it first (e.g., `ollama serve`) and ensure a model is pulled.")
    st.stop()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "eval_logger" not in st.session_state:
    st.session_state.eval_logger = EvaluationLogger()
if "interaction_count" not in st.session_state:
    st.session_state.interaction_count = 0
if "last_message_count" not in st.session_state:
    st.session_state.last_message_count = 0
if "show_dashboard" not in st.session_state:
    st.session_state.show_dashboard = False

# ADVANCED: Initialize topic classifier
if "topic_classifier" not in st.session_state:
    st.session_state.topic_classifier = TopicClassifier(llm_client=ollama, model=OLLAMA_MODEL)

# First-time welcome
if "first_visit" not in st.session_state:
    st.session_state.first_visit = True
    with st.chat_message("assistant"):
        st.markdown("""
        👋 **Welcome to Prompto!** 
        
        I can help you with:
        • 📚 Academic questions
        • 🏠 Housing & dining  
        • 💰 Financial aid
        • 🏛️ Campus services
        • 🎓 And much more!
        
        *Try asking about deadlines, locations, or campus resources!*
        """)
    st.session_state.messages.append({"role": "assistant", "content": "Welcome message shown"})

# Setup ChromaDB
force_rebuild = st.session_state.get("force_rebuild", False)
if force_rebuild:
    st.session_state.force_rebuild = False
    st.cache_resource.clear()

with st.spinner("Initializing vector DB…"):
    collection = setup_chromadb(force_rebuild=force_rebuild)

# Enhanced Sidebar
with st.sidebar:
    st.subheader("⚙️ Model & DB")
    st.write(f"**LLM:** `{OLLAMA_MODEL}`")
    st.write(f"**Embeddings:** `{'Ollama: ' + OLLAMA_EMBED_MODEL if USE_OLLAMA_EMBED else 'BAAI/bge-small-en-v1.5'}`")
    try:
        st.write(f"**Documents loaded:** {collection.count()}")
    except Exception:
        st.write("**Documents loaded:** (unknown)")

    if st.button("🔄 Rebuild Index"):
        st.session_state.force_rebuild = True
        st.cache_resource.clear()
        st.rerun()
    
    st.markdown("---")
    
    mode = st.selectbox(
        "🔍 Select Mode",
        ["Explainable AI", "Black Box"],
        help="Compare transparent AI vs traditional chatbot"
    )
    
    st.markdown("---")
    
    # ADVANCED FEATURES (Explainable Mode Only)
    if mode == "Explainable AI":
        st.subheader("🔬 Advanced Features")
        
        enable_claim_checking = st.checkbox(
            "Enable Claim-Level Checking",
            value=False,
            help="Verify each claim independently"
        )
        
        enable_policy_aware = st.checkbox(
            "Enable Policy-Aware Responses",
            value=True,
            help="Adjust responses based on topic sensitivity"
        )
        
        st.session_state.enable_claim_checking = enable_claim_checking
        st.session_state.enable_policy_aware = enable_policy_aware
        
        st.markdown("---")
    
    render_department_links()
    st.markdown("---")
    render_academic_timeline()
    st.markdown("---")
    render_emergency_contacts()
    
    if st.button("📊 View Evaluation Dashboard"):
        st.session_state.show_dashboard = True
    
    stats = st.session_state.eval_logger.get_summary_stats()
    st.markdown("### 📈 Quick Stats")
    st.write(f"**Interactions:** {stats['total_interactions']}")
    st.write(f"**Feedback:** {stats['total_feedback']}")
    if stats['total_feedback'] > 0:
        st.write(f"**Avg Trust:** {stats['avg_trust']:.1f}/5")
        st.write(f"**Helpful:** {stats['helpful_percent']:.0f}%")
    
    render_conversation_context()
    show_campus_trivia()

# Show dashboard
if st.session_state.get("show_dashboard", False):
    render_evaluation_dashboard(st.session_state.eval_logger)
    if st.button("← Back to Chat"):
        st.session_state.show_dashboard = False
        st.rerun()
    st.stop()

# Quick questions
render_quick_questions()
st.markdown("---")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            if msg["content"] != "Welcome message shown":
                format_smart_response(msg["content"], msg.get("sources", []))
            
            if msg.get("confidence"):
                conf_level = msg["confidence"]
                if conf_level == "High" or conf_level == "Very High":
                    st.caption("🟢 High Confidence")
                elif conf_level == "Medium":
                    st.caption("🟡 Medium Confidence")
                elif conf_level == "Low" or conf_level == "Very Low":
                    st.caption("🔴 Low Confidence")
                
                if msg.get("had_warning"):
                    st.caption("⚠️ Verification recommended")
            
            if msg.get("sources") and msg["content"] != "Welcome message shown":
                current_mode = "explainable" if mode == "Explainable AI" else "black_box"
                if current_mode == "explainable":
                    with st.expander(f"📚 Sources ({len(msg['sources'])}) - Click to view"):
                        st.markdown(format_sources_with_confidence(msg['sources']), unsafe_allow_html=True)
                else:
                    with st.expander(f"📚 Sources ({len(msg['sources'])})"):
                        for i, s in enumerate(msg['sources']):
                            st.markdown(f"{i+1}. [{s['title']}]({s['url']})")

# Input
prompt = st.chat_input("Ask me anything about Rutgers University…")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    latest_message_id = len(st.session_state.messages) + 1

    with st.chat_message("assistant"):
        try:
            current_mode = "explainable" if mode == "Explainable AI" else "black_box"
            enable_claim_checking = st.session_state.get("enable_claim_checking", False) and current_mode == "explainable"
            enable_policy_aware = st.session_state.get("enable_policy_aware", True)
            
            typing_animation()
            start_time = time.time()
            
            answer, sources, stream_gen, confidence_data, is_hallucination, claim_audit, policy_modulation = ask_rutgers_question(
                collection, prompt, stream=True,
                enable_claim_checking=enable_claim_checking,
                enable_policy_aware=enable_policy_aware
            )

            if stream_gen:
                placeholder = st.empty()
                buf = []
                for piece in stream_gen:
                    buf.append(piece)
                    placeholder.markdown("".join(buf))
                answer = "".join(buf)
                
                if sources:
                    is_hallucination, reason = detect_hallucination(sources, answer, confidence_data['level'])
                
                if enable_claim_checking and sources:
                    claim_verifications, claim_audit_html = perform_claim_level_checking(
                        answer, collection, llm_client=ollama, model=OLLAMA_MODEL
                    )
                    claim_audit = {'verifications': claim_verifications, 'html': claim_audit_html}
                
                if enable_policy_aware:
                    policy_modulation = apply_policy_aware_modulation(
                        prompt, answer, sources, st.session_state.topic_classifier
                    )
            
            response_time = time.time() - start_time
            st.caption(f"⚡ Response time: {response_time:.2f}s | 📚 Sources: {len(sources)}")
            
            # ADVANCED: Policy-aware disclaimers
            if current_mode == "explainable" and policy_modulation:
                risk_level = policy_modulation['risk_level']
                category = policy_modulation['category']
                
                # Debug: Show classification
                st.caption(f"🏷️ Classified as: {category} ({risk_level} risk)")
                
                if risk_level == 'high' and policy_modulation['disclaimer']:
                    st.markdown(f"""
                    <div style="background: #dc3545; color: white; padding: 1rem; 
                                border-radius: 8px; margin-bottom: 1rem; border: 3px solid #a02a2a;">
                        <div style="font-size: 1.2em; font-weight: bold; margin-bottom: 0.5rem;">
                            ⚠️ IMPORTANT SAFETY INFORMATION
                        </div>
                        <div style="white-space: pre-line;">{policy_modulation['disclaimer']}</div>
                        {f'<div style="margin-top: 0.8rem; padding-top: 0.8rem; border-top: 1px solid rgba(255,255,255,0.3); font-weight: bold;">{policy_modulation["contact_info"]}</div>' if policy_modulation.get('contact_info') else ''}
                    </div>
                    """, unsafe_allow_html=True)
                    if policy_modulation.get('modified_answer'):
                        answer = policy_modulation['modified_answer']
                elif risk_level == 'medium' and policy_modulation['disclaimer']:
                    # Show medium-risk disclaimer here instead of later
                    st.markdown(f"""
                    <div style="background: #fff3cd; border: 2px solid #ffc107; padding: 0.8rem; 
                                border-radius: 8px; margin: 1rem 0;">
                        <div style="color: #856404;">
                            <strong>⚠️ Please Verify:</strong> {policy_modulation['disclaimer']}
                        </div>
                        {f'<div style="margin-top: 0.5rem; color: #856404;"><strong>Contact:</strong> {policy_modulation["contact_info"]}</div>' if policy_modulation.get('contact_info') else ''}
                    </div>
                    """, unsafe_allow_html=True)
            
            if not stream_gen:
                format_smart_response(answer, sources)
            
            # ADVANCED: Calibrated confidence
            if current_mode == "explainable":
                st.markdown(
                    generate_confidence_badge_html(
                        confidence_data['level'], confidence_data['emoji'], confidence_data['color'], 
                        confidence_data['avg_distance'], calibrated_prob=None,
                        calibration_info=None, answer_text=answer
                    ),
                    unsafe_allow_html=True
                )
                
                if is_hallucination:
                    is_hal, reason = detect_hallucination(sources, answer, confidence_data['level'])
                    if is_hal:
                        st.markdown(generate_hallucination_warning_html(reason), unsafe_allow_html=True)
                
                # ADVANCED: Claim-level audit
                if claim_audit and claim_audit['html']:
                    with st.expander("🔍 Claim-Level Evidence Audit", expanded=False):
                        st.components.v1.html(claim_audit['html'], height=400, scrolling=True)

            # Sources
            if sources:
                if current_mode == "explainable":
                    with st.expander(f"📚 Sources ({len(sources)}) - Click to view"):
                        st.markdown(format_sources_with_confidence(sources), unsafe_allow_html=True)
                else:
                    with st.expander(f"📚 Sources ({len(sources)})"):
                        for i, s in enumerate(sources):
                            st.markdown(f"{i+1}. [{s['title']}]({s['url']})")
            
            conf_level_str = confidence_data['level'] if current_mode == "explainable" else "N/A"
            
            # Extract policy metadata
            topic_cat = None
            risk_lvl = None
            had_disclaimer = False
            if policy_modulation:
                topic_cat = policy_modulation.get('category')
                risk_lvl = policy_modulation.get('risk_level')
                had_disclaimer = policy_modulation.get('disclaimer') is not None
            
            claim_verifs = None
            if claim_audit:
                claim_verifs = claim_audit.get('verifications')
            
            # Log interaction
            st.session_state.eval_logger.log_interaction(
                question=prompt, answer=answer, sources=sources, mode=current_mode,
                confidence_level=conf_level_str,
                had_warning=is_hallucination if current_mode == "explainable" else False,
                claim_checking_enabled=enable_claim_checking,
                policy_aware_enabled=enable_policy_aware,
                topic_category=topic_cat, risk_level=risk_lvl,
                had_policy_disclaimer=had_disclaimer,
                claim_verifications=claim_verifs
            )

        except Exception as e:
            answer = f"⚠️ Error during answer generation:\n\n```\n{traceback.format_exc()}\n```"
            sources = []
            confidence_data = {'level': 'Error', 'emoji': '❌', 'color': '#dc3545', 'avg_distance': 1.0}
            conf_level_str = "Error"
            is_hallucination = True
            policy_modulation = None

    st.session_state.messages.append({
        "role": "assistant", "content": answer, "sources": sources,
        "confidence": conf_level_str if mode == "Explainable AI" else None,
        "had_warning": is_hallucination if mode == "Explainable AI" else None,
        "policy_modulation": policy_modulation if mode == "Explainable AI" else None,
        "question": prompt, "mode": current_mode, "num_sources": len(sources),
        "claim_checking_enabled": enable_claim_checking,
        "policy_aware_enabled": enable_policy_aware,
        "topic_category": topic_cat, "risk_level": risk_lvl,
        "had_policy_disclaimer": had_disclaimer,
        "message_id": latest_message_id
    })
