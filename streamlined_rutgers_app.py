#!/usr/bin/env python3
"""
Rutgers AI Assistant (Ollama edition)
- Loads Rutgers URLs with Docling
- Chunks & indexes in ChromaDB
- Answers with an Ollama-hosted LLM (GPU used automatically if available)
- Optional: use Ollama for embeddings (fully local) or SentenceTransformers

Run:
  streamlit run rutgers_ollama_app.py --server.port 8502
"""

import os
import time
import traceback
import random
from typing import List, Dict

import requests
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import base64

# --- Docling imports ---
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from docling.datamodel.document import InputDocument


# --- Ollama (LLM + optional embeddings) ---
import ollama

# --- Explainability & Evaluation modules ---
from explainability_module import (
    calculate_confidence_score,
    detect_hallucination,
    generate_confidence_badge_html,
    generate_hallucination_warning_html,
    format_sources_with_confidence
)

from evaluation_module import (
    EvaluationLogger,
    render_feedback_widget,
    render_accuracy_evaluation_widget,
    render_evaluation_dashboard
)

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# =========================
# Config (env overrides)
# =========================
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")         # e.g., "llama3.1:8b"
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
USE_OLLAMA_EMBED = os.getenv("USE_OLLAMA_EMBED", "0") == "1"    # set USE_OLLAMA_EMBED=1 to enable
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))
NUM_CTX = int(os.getenv("NUM_CTX", "8192"))                     # model context window
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
TOP_K = int(os.getenv("TOP_K", "40"))
TOP_P = float(os.getenv("TOP_P", "0.9"))
# Retrieval
N_RESULTS = int(os.getenv("N_RESULTS", "8"))
MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "4000"))     # safety cut for very large chunks
# Networking
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "25"))

# =========================
# Quick Questions & Campus Data
# =========================
# =========================
# Enhanced Quick Questions with Answers
# =========================
QUICK_QUESTIONS = {
    "What is in-state tuition?": {
        "answer": "For the 2024-2025 academic year, in-state tuition at Rutgers is approximately $16,000-$17,000 per year for full-time undergraduate students. This varies by school and program. You should check the official tuition page for the most current rates.",
        "sources": [
            {"title": "Tuition and Fees", "url": "https://admissions.rutgers.edu/costs-and-aid/tuition-fees"},
            {"title": "Student Accounting", "url": "https://finance.rutgers.edu/student-abc"}
        ]
    },
    "How to apply for financial aid?": {
        "answer": "To apply for financial aid at Rutgers, you need to:\n1. Complete the FAFSA (Free Application for Federal Student Aid)\n2. Use Rutgers school code 002629\n3. Complete any additional required forms through the Rutgers financial aid portal\n4. Submit documents by the priority deadline for maximum aid consideration",
        "sources": [
            {"title": "Financial Aid Application", "url": "https://financialaid.rutgers.edu/financial-services/apply-for-aid/how-to-apply"},
            {"title": "FAFSA Information", "url": "https://scarlethub.rutgers.edu/financial-services/apply-for-aid/how-to-apply/fafsa"}
        ]
    },
    "Where is the student center?": {
        "answer": "Rutgers has multiple student centers across campuses:\n• **College Avenue**: Rutgers Student Center\n• **Livingston**: Livingston Student Center\n• **Busch**: Busch Student Center\n• **Cook/Douglas**: Cook Student Center\nEach center offers dining, study spaces, meeting rooms, and student organization offices.",
        "sources": [
            {"title": "Student Centers", "url": "https://sca.rutgers.edu/student-centers"},
            {"title": "Campus Locations", "url": "https://newbrunswick.rutgers.edu/student-housing-and-dining"}
        ]
    },
    "What are dining hall hours?": {
        "answer": "Dining hall hours vary by location and semester. Generally:\n• **Weekdays**: 7:00 AM - 8:00 PM\n• **Weekends**: 9:00 AM - 7:00 PM\n• Some locations have extended hours\nCheck the Rutgers Food Services website for current hours and locations.",
        "sources": [
            {"title": "Dining Locations & Hours", "url": "https://food.rutgers.edu/places-eat"},
            {"title": "Meal Plan Information", "url": "https://food.rutgers.edu/meal-plans"}
        ]
    },
    "How to contact academic advising?": {
        "answer": "You can contact academic advising through:\n1. Your specific school's advising office (SAS, SOE, RBS, etc.)\n2. Schedule appointments through Starfish or your school's portal\n3. Visit advising offices during walk-in hours\n4. Email your assigned advisor directly",
        "sources": [
            {"title": "SAS Advising", "url": "https://sas.rutgers.edu/about/sas-offices/detail/office-of-advising-and-academic-services"},
            {"title": "Academic Support", "url": "https://sasundergrad.rutgers.edu/"}
        ]
    },
    "When is the add/drop period?": {
        "answer": "The add/drop period is typically the first 10 days of each semester. For Fall 2024, it's September 3-10. During this period, you can add or drop courses without penalty. Check the academic calendar for exact dates each semester.",
        "sources": [
            {"title": "Academic Calendar", "url": "https://scheduling.rutgers.edu/academic-calendar/"},
            {"title": "Registration Policies", "url": "https://nbregistrar.rutgers.edu/"}
        ]
    },
    "Where is the health center?": {
        "answer": "Rutgers Health Services has multiple locations:\n• **Hurtado Health Center** on College Avenue\n• **Busch-Livingston Health Center** \n• **Cook-Douglass Health Center**\nAll centers provide medical care, counseling, immunizations, and wellness services. Appointments are recommended.",
        "sources": [
            {"title": "Health Services Locations", "url": "https://health.rutgers.edu/about-us/hours-and-locations"},
            {"title": "Medical Services", "url": "https://health.rutgers.edu/medical-and-counseling-services/medical-services"}
        ]
    },
    "How to join a student club?": {
        "answer": "To join a student club:\n1. Browse organizations on GetInvolved.Rutgers.edu\n2. Attend the Student Involvement Fair each semester\n3. Contact club leaders through their social media or email\n4. Attend meetings and events\nThere are over 500 student organizations to choose from!",
        "sources": [
            {"title": "Student Organizations", "url": "https://sca.rutgers.edu/campus-involvement/student-organizations"},
            {"title": "Get Involved Portal", "url": "https://involvement.rutgers.edu/"}
        ]
    }
}

def render_quick_questions():
    """Render quick action buttons with pre-built answers"""
    st.markdown("### 🚀 Quick Questions")
    cols = st.columns(2)
    
    questions = list(QUICK_QUESTIONS.keys())
    for i, question in enumerate(questions):
        with cols[i % 2]:
            if st.button(f"• {question}", key=f"quick_{i}", use_container_width=True, type="secondary"):
                # Directly add to chat history like a normal message
                st.session_state.messages.append({"role": "user", "content": question})
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": QUICK_QUESTIONS[question]["answer"],
                    "sources": QUICK_QUESTIONS[question]["sources"],
                    "confidence": "High",
                    "is_quick_answer": True
                })
                st.rerun()


CAMPUS_TRIVIA = [
    "Did you know? Rutgers was founded in 1766!",
    "Fun fact: The Scarlet Knight became mascot in 1955",
    "Rutgers is the 8th oldest college in the United States",
    "Rutgers has three campuses: New Brunswick, Newark, and Camden",
    "The first intercollegiate football game was at Rutgers in 1869",
]

DEPARTMENTS = {
    "Registrar": "https://nbregistrar.rutgers.edu/",
    "Financial Aid": "https://financialaid.rutgers.edu/",
    "Housing": "https://ruoncampus.rutgers.edu/",
    "Health Services": "https://health.rutgers.edu/",
    "Career Services": "https://careers.rutgers.edu/",
    "Student Affairs": "https://studentaffairs.rutgers.edu/",
    "Academic Advising": "https://sas.rutgers.edu/academics/advising"
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
# Rutgers URLs (full list)
# =========================
RUTGERS_URLS = [
    # Registrar (Calendars, Schedules, Policies)
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

    # Financial Aid & Student Accounting
    "https://finance.rutgers.edu/student-abc/refunds",
    "https://scarlethub.rutgers.edu/financial-services/forms-documents/",
    "https://scarlethub.rutgers.edu/financial-services/financial-aid-disbursement/",
    "https://finance.rutgers.edu/student-abc",
    "https://finance.rutgers.edu/student-abc/refunds/withdrawals-school",
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

    # Housing & Dining
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

    # Student Handbook & Conduct
    "https://studentconduct.rutgers.edu/",
    "https://studentconduct.rutgers.edu/processes/university-code-student-conduct",
    "https://policies.rutgers.edu/B.aspx?BookId=11912",
    "https://studentconduct.rutgers.edu/processes",
    "https://studentconduct.rutgers.edu/about-us/contact-us",
    "https://studentconduct.rutgers.edu/faqs",
    "https://studentconduct.rutgers.edu/report-concern",
    "https://studentconduct.rutgers.edu/records-transcripts",
    "https://studentconduct.rutgers.edu/node/74",
    "https://studentconduct.rutgers.edu/node/73",
    "https://studentconduct.rutgers.edu/node/24",
    "https://studentconduct.rutgers.edu/node",
    "https://studentconduct.rutgers.edu/sites/default/files/pdf/STANDARDS-OF-CONDUCT_aug11.pdf",
    "https://studentconduct.rutgers.edu/node/72",

    # Student Life & Involvement
    "https://sca.rutgers.edu/campus-involvement/student-organizations",
    "https://sca.rutgers.edu/campus-involvement/get-involved-rutgers/getinvolvedrutgersedu",
    "https://newbrunswick.rutgers.edu/student-activities",
    "https://sca.rutgers.edu/campus-involvement",
    "https://sabo.rutgers.edu/contact",
    "https://sca.rutgers.edu/campus-involvement/student-organizations/student-organization-officers/student-organization",
    "https://comminfo.rutgers.edu/office-student-services/student-life/student-organizations",
    "https://www.business.rutgers.edu/undergraduate-new-brunswick/student-involvement",
    "https://sca.rutgers.edu/node/113",
    "https://sca.rutgers.edu/campus-involvement/get-involved-rutgers/involvement-opportunities-interest",
    "https://newbrunswick.rutgers.edu/student-experience",
    "https://myrbs.business.rutgers.edu/undergraduate-new-brunswick/student-organizations",
    "https://studentaffairs.rutgers.edu/",
    "https://studentaffairs.rutgers.edu/resources",
    "https://involvement.rutgers.edu/",
    "https://rutgers.campuslabs.com/engage/news",
    "https://recreation.rutgers.edu/club-sports",
    "https://recreation.rutgers.edu/aquatics",

    # Academics - SAS
    "https://sas.rutgers.edu/",
    "https://sas.rutgers.edu/academics",
    "https://sas.rutgers.edu/academics/majors-minors",
    "https://sas.rutgers.edu/academics/departments-programs-and-centers",
    "https://sasundergrad.rutgers.edu/",
    "https://sas.rutgers.edu/about/sas-offices/detail/office-of-advising-and-academic-services",
    "https://sas.rutgers.edu/academics/sas-divisions",
    "https://rge.sas.rutgers.edu/",
    "https://sas.rutgers.edu/students/meet-our-students",

    # Academics - SEBS
    "https://sebs.rutgers.edu/",
    "https://sebs.rutgers.edu/academics",
    "https://sebs.rutgers.edu/academics/advisors",
    "https://sebs.rutgers.edu/graduate-programs",
    "https://sebs.rutgers.edu/research",
    "https://sebs.rutgers.edu/beyond-the-classroom",
    "https://extension.rutgers.edu/",

    # Academics - SOE
    "https://soe.rutgers.edu/",
    "https://soe.rutgers.edu/academics",
    "https://soe.rutgers.edu/academics/undergraduate",
    "https://soe.rutgers.edu/academics/graduate",
    "https://soe.rutgers.edu/research/departments",
    "https://soe.rutgers.edu/research",
    "https://soe.rutgers.edu/advising",

    # Academics - Mason Gross
    "https://www.masongross.rutgers.edu/",
    "https://www.masongross.rutgers.edu/events/",
    "https://www.masongross.rutgers.edu/degrees-programs",
    "https://www.masongross.rutgers.edu/admissions",
    "https://www.masongross.rutgers.edu/calendar",
    "https://www.masongross.rutgers.edu/faculty",

    # Academics - RBS
    "https://www.business.rutgers.edu/",
    "https://www.business.rutgers.edu/undergraduate-new-brunswick",
    "https://www.business.rutgers.edu/mba",
    "https://www.business.rutgers.edu/phd",
    "https://www.business.rutgers.edu/executive-education",
    "https://www.business.rutgers.edu/faculty-research",

    # Academics - Bloustein
    "https://bloustein.rutgers.edu/",
    "https://bloustein.rutgers.edu/academics/undergraduate",
    "https://bloustein.rutgers.edu/academics/graduate",
    "https://bloustein.rutgers.edu/research",
    "https://bloustein.rutgers.edu/faculty",
    "https://bloustein.rutgers.edu/upcoming-events/",

    # Honors College
    "https://honorscollege.rutgers.edu/",
    "https://honorscollege.rutgers.edu/academics/overview",
    "https://honorscollege.rutgers.edu/academics/academic-affairs-advising",
    "https://honorscollege.rutgers.edu/academics/curriculum",
    "https://honorscollege.rutgers.edu/academics/research",
    "https://honorscollege.rutgers.edu/admissions",
    "https://honorscollege.rutgers.edu/current-students",
    "https://honorscollege.rutgers.edu/admissions/transfer-sophomores-and-juniors",

    # Health & Wellness
    "https://health.rutgers.edu/",
    "https://health.rutgers.edu/immunizations",
    "https://caps.rutgers.edu/",
    "https://health.rutgers.edu/health-education-and-promotion/health-promotion-peer-education",
    "https://health.rutgers.edu/medical-and-counseling-services/medical-services",
    "https://health.rutgers.edu/medical-and-counseling-services/counseling-services",
    "https://health.rutgers.edu/about-us/hours-and-locations",
    "https://health.rutgers.edu/medical-and-counseling-services/make-appointment",
    "https://health.rutgers.edu/medical-and-counseling-services/medical-services/sexual-health/sti-testing-treatment",
    "https://health.rutgers.edu/health-education-and-promotion/health-promotion-peer-education/workshops-and-trainings",
    "https://health.rutgers.edu/uwill",
    "https://health.rutgers.edu/resources/togetherall",

    # Title IX & Safety + Transportation
    "https://titleix.rutgers.edu/",
    "https://titleix.rutgers.edu/report",
    "https://titleix.rutgers.edu/resources",
    "https://titleix.rutgers.edu/training",
    "https://rupd.rutgers.edu/",
    "https://ipo.rutgers.edu/services",
    "https://ipo.rutgers.edu/news",
    "https://ipo.rutgers.edu/parking",
    "https://ipo.rutgers.edu/parking/permits/students",
    "https://ipo.rutgers.edu/publicsafety",
    "https://ipo.rutgers.edu/dots",
    "https://ipo.rutgers.edu/transportation",
    "https://ipo.rutgers.edu/transportation/buses/nb",
    "https://rutgers.passiogo.com/",
]

# =========================
# Embeddings
# =========================
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
    # Default: sentence-transformers (CPU ok, simple)
    # You can pin the model you've been using previously:
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-small-en-v1.5"
    )

# =========================
# LLM (Ollama)
# =========================
def call_llm(system_prompt: str, user_prompt: str, temperature: float = TEMPERATURE) -> str:
    """Single-shot generation (non-streaming) via Ollama."""
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
    """Token streaming for a nice typing effect in Streamlit."""
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

# =========================
# Data loading & chunking
# =========================
def fetch_ok(url: str, timeout: int = REQUEST_TIMEOUT) -> bool:
    """Quick HEAD to avoid long hangs on dead servers (best-effort)."""
    try:
        r = requests.head(url, timeout=timeout, allow_redirects=True)
        return r.ok
    except Exception:
        return False

def convert_urls_resilient(urls: List[str]) -> List[tuple]:
    """
    Convert each URL individually so one bad URL doesn't kill the batch.
    Returns list of (InputDocument, original_url) tuples.
    """
    docs_with_urls: List[tuple] = []
    conv = DocumentConverter()
    for i, url in enumerate(urls):
        try:
            if not fetch_ok(url):
                st.warning(f"Skipping (HEAD failed): {url}")
                continue
            res = conv.convert(url, raises_on_error=False)
            if res and res.document:
                # Store the document with its original URL
                docs_with_urls.append((res.document, url))
            else:
                st.warning(f"Docling conversion yielded no document: {url}")
        except Exception as e:
            st.warning(f"Docling convert failed for {url}: {e}")
    return docs_with_urls

def chunk_documents(docs_with_urls: List[tuple]) -> List:
    """
    Chunk documents and attach the original URL to each chunk.
    Args:
        docs_with_urls: List of (InputDocument, original_url) tuples
    Returns:
        List of (chunk, original_url) tuples
    """
    chunker = HybridChunker(
        max_tokens=8191,          # tokenizer handled internally; generous
        merge_peers=True,
    )
    all_chunks = []
    for doc, original_url in docs_with_urls:
        try:
            for ch in chunker.chunk(dl_doc=doc):
                # Safety: cap chunk text length to avoid overlong contexts downstream
                if hasattr(ch, "text"):
                    ch.text = ch.text[:MAX_CHUNK_CHARS]
                # Attach the original URL directly to each chunk
                all_chunks.append((ch, original_url))
        except Exception as e:
            st.warning(f"Chunking failed: {e}")
    return all_chunks

def build_metadata(chunk, url_fallback: str):
    title = "Rutgers Document"
    filename = "doc"
    try:
        # try to get better title
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

# =========================
# Vector DB (Chroma)
# =========================
@st.cache_resource
def setup_chromadb(force_rebuild: bool = False):
    client = chromadb.PersistentClient(path="./chroma_db")
    emb_fn = get_embedding_function()

    # Check if we need to rebuild
    if not force_rebuild:
        try:
            col = client.get_collection("rutgers_docs", embedding_function=emb_fn)
            if col.count() > 0:
                return col
        except Exception:
            pass

    # Delete existing collection if it exists (for rebuild)
    try:
        client.delete_collection("rutgers_docs")
    except Exception:
        pass

    col = client.create_collection(
        name="rutgers_docs",
        metadata={"description": "Rutgers University documents and information"},
        embedding_function=emb_fn,
    )

    # Load data
    with st.spinner("Converting Rutgers URLs with Docling…"):
        docs_with_urls = convert_urls_resilient(RUTGERS_URLS)

    with st.spinner("Chunking documents…"):
        chunks = chunk_documents(docs_with_urls)

    # Prepare rows
    documents, ids, metadatas = [], [], []
    for i, item in enumerate(chunks):
        # item is (chunk, original_url) per chunk_documents()
        try:
            ch, original_url = item
        except Exception:
            # Defensive fallback
            ch = item
            original_url = ""

        txt = getattr(ch, "text", "")
        if not txt.strip():
            continue
        
        # Try to extract URL from chunk metadata first, fallback to original_url
        url = original_url  # Default to the URL we tracked
        try:
            if hasattr(ch, "meta") and hasattr(ch.meta, "origin"):
                # Check various possible source fields
                if hasattr(ch.meta.origin, "source") and ch.meta.origin.source:
                    origin_src = ch.meta.origin.source
                    url = origin_src if isinstance(origin_src, str) else str(origin_src)
                elif hasattr(ch.meta.origin, "uri") and ch.meta.origin.uri:
                    url = str(ch.meta.origin.uri)
                elif hasattr(ch.meta.origin, "url") and ch.meta.origin.url:
                    url = str(ch.meta.origin.url)
        except Exception:
            pass  # Keep the original_url fallback

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
    return placeholder

def format_smart_response(answer, sources):
    """Detect response type and format accordingly"""
    
    # Detect urgent information (health, safety, deadlines)
    urgent_keywords = ['emergency', 'deadline', 'urgent', 'crisis', 'immediately']
    if any(keyword in answer.lower() for keyword in urgent_keywords):
        st.markdown('<div class="urgent-message">🚨 ' + answer + '</div>', unsafe_allow_html=True)
    
    # Detect step-by-step instructions
    elif 'step' in answer.lower() or 'first' in answer.lower() and 'then' in answer.lower():
        steps = answer.split('\n')
        st.markdown("### 📋 Step-by-Step Guide:")
        for step in steps:
            if step.strip():
                st.markdown(f"• {step.strip()}")
    
    else:
        # Regular formatted response
        st.markdown(answer)

def render_quick_questions():
    """Render quick action buttons with pre-built answers"""
    st.markdown("### 🚀 Quick Questions")
    cols = st.columns(2)
    
    questions = list(QUICK_QUESTIONS.keys())
    for i, question in enumerate(questions):
        with cols[i % 2]:
            if st.button(f"• {question}", key=f"quick_{i}", use_container_width=True, type="secondary"):
                # Directly add to chat history like a normal message
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
    """Render department quick links in sidebar"""
    st.sidebar.markdown("### 📞 Quick Departments")
    for dept, url in DEPARTMENTS.items():
        st.sidebar.markdown(f'<a href="{url}" target="_blank" style="color: white; text-decoration: none;">📎 {dept}</a>', 
                           unsafe_allow_html=True)

def render_academic_timeline():
    """Render academic timeline in sidebar"""
    st.sidebar.markdown("### 📅 Academic Timeline")
    for event, date in IMPORTANT_DATES.items():
        st.sidebar.write(f"**{event}:** {date}")

def render_emergency_contacts():
    """Render emergency contacts in sidebar"""
    if st.sidebar.button("🚨 Emergency Contacts"):
            st.markdown("""
            **Campus Police:** (848) 932-7211  
            **Health Services:** (848) 932-7402  
            **Counseling:** (848) 932-7884  
            **Title IX:** (848) 932-8200
            **Any Emergency:** 911
            """)

def render_response_rating():
    """Add response rating system after each assistant response"""
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("👍 Helpful", key="helpful", use_container_width=True):
                st.success("Thanks for your feedback!")
                # Log positive feedback
                st.session_state.eval_logger.log_feedback(
                    question=st.session_state.messages[-2]["content"] if len(st.session_state.messages) > 1 else "Unknown",
                    rating=5,
                    comments="User marked as helpful"
                )
        with col2:
            if st.button("👎 Not Helpful", key="not_helpful", use_container_width=True):
                st.info("Sorry about that! Try rephrasing your question.")
        with col3:
            if st.button("📋 More Details", key="more_details", use_container_width=True):
                with st.expander("Need more help?"):
                    st.markdown("""
                    **Contact Rutgers Support:**
                    - 📞 General Help: (848) 445-INFO
                    - 🌐 [Rutgers Help Portal](https://myrun.newark.rutgers.edu/)
                    - 💻 [IT Help Desk](https://it.rutgers.edu/)
                    """)

def render_conversation_context():
    """Show recent conversation topics in sidebar"""
    if len(st.session_state.messages) > 4:
        st.sidebar.markdown("### 💭 Conversation Context")
        st.sidebar.caption("Recent topics:")
        recent_topics = [msg["content"][:30] + "..." for msg in st.session_state.messages[-3:] if msg["role"] == "user"]
        for topic in recent_topics:
            st.sidebar.write(f"• {topic}")

def show_campus_trivia():
    """Show campus trivia every 2 interactions (works for both quick questions and regular chat)"""
    # Count all interactions (both quick questions and regular chat)
    if "interaction_count" not in st.session_state:
        st.session_state.interaction_count = 0
    
    # Check if we have a new interaction
    total_messages = len([msg for msg in st.session_state.messages 
                         if msg["content"] != "Welcome message shown"])
    
    # Only count when we have a new pair of messages (user + assistant)
    if total_messages > st.session_state.last_message_count:
        # Check if this is a complete interaction (user message followed by assistant)
        if (len(st.session_state.messages) >= 2 and 
            st.session_state.messages[-1]["role"] == "assistant" and 
            st.session_state.messages[-2]["role"] == "user"):
            
            st.session_state.interaction_count += 1
            st.session_state.last_message_count = total_messages
            
            # Show trivia every 2 interactions
            if st.session_state.interaction_count % 2 == 0:
                st.sidebar.markdown("---")
                st.sidebar.markdown("### 🎓 Campus Trivia")
                st.sidebar.markdown(f"*{random.choice(CAMPUS_TRIVIA)}*")
                st.sidebar.markdown("")

# =========================
# RAG ask function
# =========================
def ask_rutgers_question(col, question: str, stream: bool = True):
    """
    Enhanced RAG with explainability features.
    Returns: (answer, chunks, stream_generator, confidence_level, is_hallucination)
    """
    chunks = search_semantic(col, question, n_results=N_RESULTS)
    
    # Calculate confidence early
    confidence_level, emoji, color, avg_distance = calculate_confidence_score(chunks)
    
    if not chunks:
        return "Sorry, I couldn't find relevant information about that topic in the Rutgers data.", [], None, "No Data", True

    parts = []
    for ch in chunks:
        # keep prompt compact; model quality > verbatim wall of text
        snippet = ch["text"]
        parts.append(f"[{ch['title']}]({ch['url']})\n{snippet}")

    context = "\n\n---\n\n".join(parts[:N_RESULTS])

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
        return None, chunks, call_llm_stream(system, user), confidence_level, False
    else:
        answer = call_llm(system, user)
        
        # Detect potential hallucination
        is_hallucination, reason = detect_hallucination(chunks, answer, confidence_level)
        
        return answer, chunks, None, confidence_level, is_hallucination

# =========================
# UI
# =========================
st.set_page_config(page_title="Prompto", layout="wide")

st.markdown("""
<style>
/* Overall Page Background - Subtle texture or solid light */
.stApp {
    background: linear-gradient(135deg, #8B0000 0%, #A52A2A 25%, #B22222, #CD5C5C 75%, #DC143C 100%);
    background-size: 400% 400%; 
    animation: gradientShift 15s ease infinite;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    min-height: 100vh;
}

/* Header Container with image overlay */
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

/* Logo Styling - Positioned above text */
.logo-container {
    position: relative;
    z-index: 2;
    margin-bottom: -20px; /* Overlap with text */
}

/* Title Styling - Positioned below image */
.title-container {
    position: relative;
    z-index: 1;
    margin-top: -10px; /* Pull text up to overlap with image */
    padding-top: 40px; /* Space for image */
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

/* Scarlet Divider */
.scarlet-divider {
    height: 4px;
    background: linear-gradient(90deg, #CC0000, #8B0000, #CC0000);
    margin: 10px auto 20px auto;
    width: 80%;
    border-radius: 2px;
}

/* Enhanced Sidebar Styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #CC0000 0%, #8B0000 100%);
    color: white;
    padding-top: 2rem;
}

[data-testid="stSidebar"] .sidebar-header {
    text-align: center;
    padding: 0 0 20px 0;
    border-bottom: 2px solid rgba(255,255,255,0.2);
    margin-bottom: 20px;
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

/* Enhanced Buttons */
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

/* Enhanced Chat Input */
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

[data-testid="stTextInput"] > div > button {
    background: linear-gradient(135deg, #CC0000, #A00000);
    color: white;
    border-radius: 12px;
    padding: 10px 20px;
    transition: all 0.3s ease;
    border: none;
}

[data-testid="stTextInput"] > div > button:hover {
    background: linear-gradient(135deg, #FF3333, #CC0000);
    transform: scale(1.05);
}

/* Enhanced Chat Messages */
[data-testid="stChatMessage"] {
    background: white;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    margin-bottom: 1.2rem;
    padding: 1rem 1.5rem;
    border: 1px solid #e0e0e0;
    transition: all 0.3s ease;
}

[data-testid="stChatMessage"]:hover {
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
    transform: translateY(-2px);
}

/* User Message specific styling */
[data-testid="stChatMessage"]:has(div.st-emotion-cache-p5mllp) {
    background: linear-gradient(135deg, #f0f0f0, #e8e8e8);
    text-align: right;
    border-top-right-radius: 0;
    border-left: 4px solid #CC0000;
}

[data-testid="stChatMessage"]:has(div.st-emotion-cache-1r7r3cn) {
    background: linear-gradient(135deg, #ffffff, #f8f8f8);
    text-align: left;
    border-top-left-radius: 0;
    border-left: 4px solid #CC0000;
}

/* Custom Message Bubbles */
.user-message {
    background: linear-gradient(135deg, #CC0000, #8B0000);
    color: white;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 16px;
    margin: 8px 0;
    max-width: 80%;
    margin-left: auto;
}

.assistant-message {
    background: linear-gradient(135deg, #f8f9fa, #e9ecef);
    color: #333;
    border-radius: 18px 18px 18px 4px;
    padding: 12px 16px;
    margin: 8px 0;
    max-width: 80%;
    border-left: 4px solid #CC0000;
}

.urgent-message {
    background: linear-gradient(135deg, #fff3cd, #ffeeba);
    border: 2px solid #ffc107;
    border-radius: 12px;
    padding: 12px;
    margin: 10px 0;
    color: #856404;
    font-weight: bold;
}

/* Source Box - Enhanced */
.source-box { 
    background: linear-gradient(135deg, #FFF5F5, #FFE8E8);
    padding: 1rem; 
    border-radius: 10px; 
    margin: .6rem 0; 
    border-left: 6px solid #CC0000;
    color: #333; 
    box-shadow: 0 2px 8px rgba(204,0,0,0.1);
    transition: all 0.3s ease;
}

.source-box:hover {
    transform: translateX(5px);
    box-shadow: 0 4px 12px rgba(204,0,0,0.2);
}

.source-box a { 
    color: #990000;
    text-decoration: none; 
    font-weight: bold; 
    transition: color 0.2s;
}
.source-box a:hover {
    color: #CC0000;
}

/* Confidence Badges - Enhanced */
.confidence-badge {
    display: inline-block;
    padding: 8px 16px;
    border-radius: 20px;
    font-weight: bold;
    margin: 10px 5px 5px 0;
    font-size: 0.85em;
    background: linear-gradient(135deg, #f8f9fa, #e9ecef);
    color: #333;
    border: 2px solid #ddd;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.confidence-badge.high { 
    background: linear-gradient(135deg, #d4edda, #c3e6cb);
    color: #155724; 
    border-color: #28a745;
}
.confidence-badge.medium { 
    background: linear-gradient(135deg, #fff3cd, #ffeeba);
    color: #856404; 
    border-color: #ffc107;
}
.confidence-badge.low { 
    background: linear-gradient(135deg, #f8d7da, #f5c6cb);
    color: #721c24; 
    border-color: #dc3545;
}

.warning-badge {
    display: inline-block;
    padding: 8px 16px;
    border-radius: 20px;
    font-weight: bold;
    margin: 10px 5px;
    background: linear-gradient(135deg, #ffebee, #ffcdd2);
    color: #CC0000;
    border: 2px solid #CC0000;
    font-size: 0.85em;
    box-shadow: 0 2px 8px rgba(204,0,0,0.2);
}

/* Quick Questions Styling */
.quick-question-btn {
    background: linear-gradient(135deg, #CC0000, #A00000) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 15px !important;
    margin: 5px 0 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 2px 8px rgba(204,0,0,0.2) !important;
    width: 100% !important;
}

.quick-question-btn:hover {
    background: linear-gradient(135deg, #FF3333, #CC0000) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 15px rgba(255,51,51,0.3) !important;
}

/* Stats Card */
.stats-card {
    background: white;
    padding: 15px;
    border-radius: 12px;
    border-left: 5px solid #CC0000;
    margin: 10px 0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
}

.stats-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.15);
}

/* Floating animation for header */
@keyframes float {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-5px); }
    100% { transform: translateY(0px); }
}

.logo-container img {
    animation: float 3s ease-in-out infinite;
}

/* Pulse animation for new messages */
@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.02); }
    100% { transform: scale(1); }
}

.new-message {
    animation: pulse 0.5s ease-in-out;
}
            
.header-logo {
    width: 100px;
    border-radius: 50%;
    box-shadow: 0 8px 25px rgba(204,0,0,0.3);
    border: 4px solid white;
}

/* Rating buttons */
.rating-buttons {
    display: flex;
    gap: 10px;
    margin-top: 15px;
    padding-top: 15px;
    border-top: 1px solid #eee;
}

/* Background animation */
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
</style>
""", unsafe_allow_html=True)

# Header with image
try:
    img_base64 = get_base64_image("scarlet_logo.png")
    st.markdown(f"""
    <div class="header-container">
        <div class="logo-container">
            <img src='data:image/png;base64,{img_base64}' class='header-logo' alt='Scarlet Mind Logo'>
        </div>
        <div class="title-container">
            <h1 class="main-title">Prompto</h1>
            <p class="main-subtitle">Your AI Assistant for Rutgers University</p>
        </div>
        <div class="scarlet-divider"></div>
    </div>
    """, unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("Logo image not found.")
    st.markdown("""
    <div class="header-container">
        <div class="title-container">
            <h1 class="main-title">Prompto</h1>
            <p class="main-subtitle">Your AI Assistant for Rutgers University</p>
        </div>
        <div class="scarlet-divider"></div>
    </div>
    """, unsafe_allow_html=True)

# Quick Ollama health check
try:
    _ = ollama.list()  # ensures daemon is reachable
except Exception:
    st.error("Could not reach Ollama. Start it first (e.g., `ollama serve`) and ensure a model is pulled.")
    st.stop()

# Cache chat state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize evaluation logger
if "eval_logger" not in st.session_state:
    st.session_state.eval_logger = EvaluationLogger()

# Initialize trivia tracking
if "interaction_count" not in st.session_state:
    st.session_state.interaction_count = 0
if "last_message_count" not in st.session_state:
    st.session_state.last_message_count = 0

# Initialize dashboard state
if "show_dashboard" not in st.session_state:
    st.session_state.show_dashboard = False

# First-time user experience
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
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "Welcome message shown"
    })

# Setup Chroma
# Force rebuild flag (can be triggered by sidebar button)
force_rebuild = st.session_state.get("force_rebuild", False)
if force_rebuild:
    st.session_state.force_rebuild = False  # Reset flag
    st.cache_resource.clear()  # Clear cache to force rebuild

with st.spinner("Initializing vector DB…"):
    collection = setup_chromadb(force_rebuild=force_rebuild)

# Sidebar info
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
    
    # Mode selector
    mode = st.selectbox(
        "🔍 Select Mode",
        ["Explainable AI", "Black Box"],
        help="Compare transparent AI vs traditional chatbot"
    )
    
    st.markdown("---")
    
    # Enhanced sidebar features
    render_department_links()
    st.markdown("---")
    render_academic_timeline()
    st.markdown("---")
    render_emergency_contacts()
        
    
    # Evaluation dashboard button
    if st.button("📊 View Evaluation Dashboard"):
        st.session_state.show_dashboard = True
    
    # Show stats preview
    stats = st.session_state.eval_logger.get_summary_stats()
    st.markdown("### 📈 Quick Stats")
    st.write(f"**Interactions:** {stats['total_interactions']}")
    st.write(f"**Feedback:** {stats['total_feedback']}")
    if stats['total_feedback'] > 0:
        st.write(f"**Avg Trust:** {stats['avg_trust']:.1f}/5")
        st.write(f"**Helpful:** {stats['helpful_percent']:.0f}%")
    
    # Show conversation context and trivia
    render_conversation_context()
    show_campus_trivia()

# Show dashboard if requested
if st.session_state.get("show_dashboard", False):
    render_evaluation_dashboard(st.session_state.eval_logger)
    if st.button("← Back to Chat"):
        st.session_state.show_dashboard = False
        st.rerun()
    st.stop()

# Render quick questions
render_quick_questions()
st.markdown("---")


# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(f'<div class="user-message">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            if msg["content"] != "Welcome message shown":
                format_smart_response(msg["content"], msg.get("sources", []))         
            # Show confidence badge for explainable mode answers
            if msg["role"] == "assistant" and msg.get("confidence"):
                conf_level = msg["confidence"]
                if conf_level == "High":
                    st.caption("🟢 High Confidence")
                elif conf_level == "Medium":
                    st.caption("🟡 Medium Confidence")
                elif conf_level == "Low":
                    st.caption("🔴 Low Confidence")
                
                # Show warning badge if applicable
                if msg.get("had_warning"):
                    st.caption("⚠️ Verification recommended")
            
            # Show sources
            if msg.get("sources") and msg["content"] != "Welcome message shown":
                current_mode = "explainable" if mode == "Explainable AI" else "black_box"
                if current_mode == "explainable":
                    with st.expander(f"📚 Sources ({len(msg['sources'])}) - Click to view"):
                        st.markdown(
                            format_sources_with_confidence(msg['sources']),
                            unsafe_allow_html=True
                        )
                else:
                    # Black box mode: minimal source display
                    with st.expander(f"📚 Sources ({len(msg['sources'])})"):
                        for i, s in enumerate(msg['sources']):
                            st.markdown(f"{i+1}. [{s['title']}]({s['url']})")

# Input
prompt = st.chat_input("Ask me anything about Rutgers University…")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(f'<div class="user-message">{prompt}</div>', unsafe_allow_html=True)

    with st.chat_message("assistant"):
        try:
            # Determine current mode
            current_mode = "explainable" if mode == "Explainable AI" else "black_box"
            
            # Show typing animation
            placeholder = typing_animation()
            
            # Get response
            answer, sources, stream_gen, confidence_level, is_hallucination = ask_rutgers_question(
                collection, prompt, stream=True
            )

            # Clear typing animation
            placeholder.empty()

            # Stream response
            if stream_gen:
                response_placeholder = st.empty()
                buf = []
                start_time = time.time()
                
                for piece in stream_gen:
                    buf.append(piece)
                    response_placeholder.markdown("".join(buf))
                answer = "".join(buf)
                
                # Calculate response time
                response_time = time.time() - start_time
                
                # Re-check for hallucination after streaming
                if sources:
                    is_hallucination, reason = detect_hallucination(sources, answer, confidence_level)
            else:
                start_time = time.time()
                format_smart_response(answer, sources)
                response_time = time.time() - start_time
            
            # Show performance metrics
            st.caption(f"⚡ Response time: {response_time:.2f}s | 📚 Sources: {len(sources)}")
            
            # EXPLAINABLE MODE: Show confidence and warnings
            if current_mode == "explainable":
                # Calculate confidence
                conf_level, conf_emoji, conf_color, avg_dist = calculate_confidence_score(sources)
                
                # Display confidence badge
                st.markdown(
                    generate_confidence_badge_html(conf_level, conf_emoji, conf_color, avg_dist),
                    unsafe_allow_html=True
                )
                
                # Display hallucination warning if detected
                if is_hallucination:
                    is_hal, reason = detect_hallucination(sources, answer, conf_level)
                    if is_hal:
                        st.markdown(
                            generate_hallucination_warning_html(reason),
                            unsafe_allow_html=True
                        )

            # Show sources (both modes, but formatted differently)
            if sources:
                if current_mode == "explainable":
                    with st.expander(f"📚 Sources ({len(sources)}) - Click to view"):
                        st.markdown(
                            format_sources_with_confidence(sources),
                            unsafe_allow_html=True
                        )
                else:
                    # Black box mode: minimal source display
                    with st.expander(f"📚 Sources ({len(sources)})"):
                        for i, s in enumerate(sources):
                            st.markdown(f"{i+1}. [{s['title']}]({s['url']})")
            
            # Log interaction
            st.session_state.eval_logger.log_interaction(
                question=prompt,
                answer=answer,
                sources=sources,
                mode=current_mode,
                confidence_level=confidence_level if current_mode == "explainable" else "N/A",
                had_warning=is_hallucination if current_mode == "explainable" else False
            )
            
            # Render feedback widget
            render_feedback_widget(
                logger=st.session_state.eval_logger,
                question=prompt,
                mode=current_mode,
                confidence_level=confidence_level if current_mode == "explainable" else "N/A",
                num_sources=len(sources),
                had_warning=is_hallucination if current_mode == "explainable" else False
            )
            
            # Render accuracy evaluation (for testers)
            render_accuracy_evaluation_widget(
                logger=st.session_state.eval_logger,
                question=prompt,
                answer=answer
            )
            
            # Add response rating
            render_response_rating()

        except Exception as e:
            answer = f"⚠️ Error during answer generation:\n\n```\n{traceback.format_exc()}\n```"
            sources = []
            confidence_level = "Error"
            is_hallucination = True

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "confidence": confidence_level if mode == "Explainable AI" else None,
        "had_warning": is_hallucination if mode == "Explainable AI" else None
    })