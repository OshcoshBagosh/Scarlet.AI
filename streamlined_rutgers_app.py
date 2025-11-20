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

# --- Explainability & Evaluation modules ---
from explainability_module import (
    calculate_confidence_score,
    detect_hallucination,
    generate_confidence_badge_html,
    generate_hallucination_warning_html,
    format_sources_with_confidence,
    load_calibration_model,
    perform_claim_level_checking
)

from evaluation_module import (
    EvaluationLogger,
    render_feedback_widget,
    render_accuracy_evaluation_widget,
    render_evaluation_dashboard
)

# --- Topic Classification & Policy-Aware Response ---
from topic_classifier import TopicClassifier
from policy_aware_module import (
    PolicyAwareResponder,
    create_risk_aware_rag_prompt,
    apply_policy_aware_modulation
)

# =========================
# Config (env overrides)
# =========================
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")         # e.g., "llama3.1:8b"
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
    # You can pin the model you’ve been using previously:
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
# RAG ask function
# =========================
def ask_rutgers_question(col, question: str, stream: bool = True, 
                        enable_claim_checking: bool = False, 
                        enable_policy_aware: bool = True):
    """
    Enhanced RAG with explainability features, claim-level checking, and policy-aware responses.
    Returns: (answer, chunks, stream_generator, confidence_data, is_hallucination, claim_audit, policy_modulation)
    """
    # Step 1: Classify topic for policy-aware routing
    topic_classification = None
    policy_modulation = None
    
    if enable_policy_aware:
        if 'topic_classifier' not in st.session_state:
            st.session_state.topic_classifier = TopicClassifier(llm_client=ollama, model=OLLAMA_MODEL)
        
        topic_classification = st.session_state.topic_classifier.classify(question, use_llm=False)
    
    # Step 2: Retrieve relevant chunks
    chunks = search_semantic(col, question, n_results=N_RESULTS)
    
    # Step 3: Calculate calibrated confidence
    confidence_level, emoji, color, avg_distance, calibrated_prob, calibration_info = calculate_confidence_score(
        chunks, use_calibration=True
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

    # Step 4: Create policy-aware system prompt if enabled
    if enable_policy_aware and topic_classification:
        guidelines = st.session_state.topic_classifier.get_risk_guidelines(
            topic_classification['category_enum']
        )
        
        # Add risk-specific instructions to prompt
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

    # Step 5: Generate answer
    if stream:
        return None, chunks, call_llm_stream(system, user), confidence_data, False, None, topic_classification
    else:
        answer = call_llm(system, user)
        
        # Step 6: Detect potential hallucination
        is_hallucination, reason = detect_hallucination(chunks, answer, confidence_level)
        
        # Step 7: Perform claim-level checking if enabled
        claim_audit = None
        if enable_claim_checking:
            claim_verifications, claim_audit_html = perform_claim_level_checking(
                answer, col, llm_client=ollama, model=OLLAMA_MODEL
            )
            claim_audit = {
                'verifications': claim_verifications,
                'html': claim_audit_html
            }
        
        # Step 8: Apply policy-aware modulation
        if enable_policy_aware and topic_classification:
            policy_modulation = apply_policy_aware_modulation(
                question, answer, chunks, st.session_state.topic_classifier
            )
        
        return answer, chunks, None, confidence_data, is_hallucination, claim_audit, policy_modulation

# =========================
# UI
# =========================
st.set_page_config(page_title="Rutgers AI Assistant (Ollama)", page_icon="🏛️", layout="wide")

st.markdown("""
<style>
.main-header { color:#CC0000; text-align:center; font-size:2.2em; margin-bottom:0.7em; }
.source-box { background:#f9f9f9; padding:.6rem; border-radius:6px; margin:.4rem 0; border-left:3px solid #CC0000; color:#000; }
.source-box a { color:#CC0000; text-decoration:none; font-weight:bold; }
.small { font-size: 0.9em; color:#333; }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🏛️ Rutgers University AI Assistant (Ollama)</h1>', unsafe_allow_html=True)

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

# Initialize dashboard state
if "show_dashboard" not in st.session_state:
    st.session_state.show_dashboard = False

# Load calibration model on startup
if "calibration_loaded" not in st.session_state:
    try:
        if load_calibration_model():
            st.session_state.calibration_loaded = True
            st.sidebar.success("✅ Calibration model loaded")
        else:
            st.session_state.calibration_loaded = False
            st.sidebar.info("ℹ️ No calibration model found. Run calibration_script.py to create one.")
    except Exception as e:
        st.session_state.calibration_loaded = False
        st.sidebar.warning(f"⚠️ Calibration model load failed: {e}")

# Initialize topic classifier
if "topic_classifier" not in st.session_state:
    st.session_state.topic_classifier = TopicClassifier(llm_client=ollama, model=OLLAMA_MODEL)

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
    
    # Advanced Features (Explainable Mode Only)
    if mode == "Explainable AI":
        st.subheader("🔬 Advanced Features")
        
        enable_claim_checking = st.checkbox(
            "Enable Claim-Level Checking",
            value=False,
            help="Verify each claim in the answer independently"
        )
        
        enable_policy_aware = st.checkbox(
            "Enable Policy-Aware Responses",
            value=True,
            help="Adjust responses based on topic sensitivity"
        )
        
        # Store in session state
        st.session_state.enable_claim_checking = enable_claim_checking
        st.session_state.enable_policy_aware = enable_policy_aware
        
        st.markdown("---")
    
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

# Show dashboard if requested
if st.session_state.get("show_dashboard", False):
    render_evaluation_dashboard(st.session_state.eval_logger)
    if st.button("← Back to Chat"):
        st.session_state.show_dashboard = False
        st.rerun()
    st.stop()

"""Chat history rendering (without feedback forms to avoid delayed appearance)."""
# Render past chat messages (excluding feedback forms; feedback shown only for latest answer)
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("confidence"):
            conf_level = msg["confidence"]
            if conf_level == "High":
                st.caption("🟢 High Confidence")
            elif conf_level == "Medium":
                st.caption("🟡 Medium Confidence")
            elif conf_level == "Low":
                st.caption("🔴 Low Confidence")
            if msg.get("had_warning"):
                st.caption("⚠️ Verification recommended")
        if msg.get("sources"):
            with st.expander(f"📚 Sources ({len(msg['sources'])})"):
                for i, s in enumerate(msg["sources"]):
                    url = s["url"]
                    distance = s.get("distance", 1.0)
                    st.markdown(
                        f"""
                        <div class="source-box">
                          <strong>{i+1}. {s['title']}</strong><br>
                          <span class="small">Relevance: {distance:.3f} |
                          <a href="{url}" target="_blank">🔗 View Source</a></span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

# Input
prompt = st.chat_input("Ask me anything about Rutgers University…")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    # Pre-compute prospective assistant message id for stable feedback widget
    latest_message_id = len(st.session_state.messages) + 1

    with st.chat_message("assistant"):
        try:
            # Determine current mode
            current_mode = "explainable" if mode == "Explainable AI" else "black_box"
            
            # Get feature flags
            enable_claim_checking = st.session_state.get("enable_claim_checking", False) and current_mode == "explainable"
            enable_policy_aware = st.session_state.get("enable_policy_aware", True)
            
            with st.spinner("Searching Rutgers knowledge base…"):
                answer, sources, stream_gen, confidence_data, is_hallucination, claim_audit, policy_modulation = ask_rutgers_question(
                    collection, prompt, stream=True,
                    enable_claim_checking=enable_claim_checking,
                    enable_policy_aware=enable_policy_aware
                )

            # Stream response
            if stream_gen:
                placeholder = st.empty()
                buf = []
                for piece in stream_gen:
                    buf.append(piece)
                    placeholder.markdown("".join(buf))
                answer = "".join(buf)
                
                # Re-check for hallucination after streaming
                if sources:
                    is_hallucination, reason = detect_hallucination(sources, answer, confidence_data['level'])
                
                # Re-do claim checking and policy modulation after streaming
                if enable_claim_checking and sources:
                    claim_verifications, claim_audit_html = perform_claim_level_checking(
                        answer, collection, llm_client=ollama, model=OLLAMA_MODEL
                    )
                    claim_audit = {
                        'verifications': claim_verifications,
                        'html': claim_audit_html
                    }
                
                if enable_policy_aware:
                    policy_modulation = apply_policy_aware_modulation(
                        prompt, answer, sources, st.session_state.topic_classifier
                    )
            
            # POLICY-AWARE MODE: Show critical disclaimers at top
            if current_mode == "explainable" and policy_modulation:
                risk_level = policy_modulation['risk_level']
                
                # High-risk topics get prominent disclaimer at top
                if risk_level == 'high' and policy_modulation['disclaimer']:
                    st.markdown(f"""
                    <div style="background: #dc3545; color: white; padding: 1rem; 
                                border-radius: 8px; margin-bottom: 1rem; border: 3px solid #a02a2a;">
                        <div style="font-size: 1.2em; font-weight: bold; margin-bottom: 0.5rem;">
                            ⚠️ IMPORTANT SAFETY INFORMATION
                        </div>
                        <div style="white-space: pre-line;">
                            {policy_modulation['disclaimer']}
                        </div>
                        {f'<div style="margin-top: 0.8rem; padding-top: 0.8rem; border-top: 1px solid rgba(255,255,255,0.3); font-weight: bold;">{policy_modulation["contact_info"]}</div>' if policy_modulation.get('contact_info') else ''}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Use modulated answer for high-risk topics
                    if policy_modulation.get('modified_answer'):
                        answer = policy_modulation['modified_answer']
            
            # Display answer (potentially modulated)
            if not stream_gen:
                st.markdown(answer)
            
            # EXPLAINABLE MODE: Show confidence and warnings
            if current_mode == "explainable":
                # Display calibrated confidence badge
                st.markdown(
                    generate_confidence_badge_html(
                        confidence_data['level'],
                        confidence_data['emoji'],
                        confidence_data['color'],
                        confidence_data['avg_distance'],
                        confidence_data.get('calibrated_prob'),
                        confidence_data.get('calibration_info'),
                        answer_text=answer
                    ),
                    unsafe_allow_html=True
                )
                
                # Display hallucination warning if detected
                if is_hallucination:
                    is_hal, reason = detect_hallucination(sources, answer, confidence_data['level'])
                    if is_hal:
                        st.markdown(
                            generate_hallucination_warning_html(reason),
                            unsafe_allow_html=True
                        )
                
                # Medium-risk disclaimer (shown after answer)
                if policy_modulation and policy_modulation['risk_level'] == 'medium' and policy_modulation['disclaimer']:
                    st.markdown(f"""
                    <div style="background: #fff3cd; border: 2px solid #ffc107; padding: 0.8rem; 
                                border-radius: 8px; margin: 1rem 0;">
                        <div style="color: #856404;">
                            <strong>⚠️ Please Verify:</strong> {policy_modulation['disclaimer']}
                        </div>
                        {f'<div style="margin-top: 0.5rem; color: #856404;"><strong>Contact:</strong> {policy_modulation["contact_info"]}</div>' if policy_modulation.get('contact_info') else ''}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show claim-level audit if enabled
                if claim_audit and claim_audit['html']:
                    with st.expander("🔍 Claim-Level Evidence Audit", expanded=False):
                        st.components.v1.html(claim_audit['html'], height=400, scrolling=True)

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
            
            # Extract confidence level for logging
            conf_level_str = confidence_data['level'] if current_mode == "explainable" else "N/A"
            
            # Extract policy-aware metadata
            topic_cat = None
            risk_lvl = None
            had_disclaimer = False
            if policy_modulation:
                topic_cat = policy_modulation.get('category')
                risk_lvl = policy_modulation.get('risk_level')
                had_disclaimer = policy_modulation.get('disclaimer') is not None
            
            # Extract claim verifications
            claim_verifs = None
            if claim_audit:
                claim_verifs = claim_audit.get('verifications')
            
            # Log interaction with all feature metadata
            st.session_state.eval_logger.log_interaction(
                question=prompt,
                answer=answer,
                sources=sources,
                mode=current_mode,
                confidence_level=conf_level_str,
                had_warning=is_hallucination if current_mode == "explainable" else False,
                claim_checking_enabled=enable_claim_checking,
                policy_aware_enabled=enable_policy_aware,
                topic_category=topic_cat,
                risk_level=risk_lvl,
                had_policy_disclaimer=had_disclaimer,
                claim_verifications=claim_verifs
            )

            # Render feedback widget immediately for this answer (before rerun)
            latest_message_id = len(st.session_state.messages) + 1  # prospective id
            render_feedback_widget(
                logger=st.session_state.eval_logger,
                question=prompt,
                mode=current_mode,
                confidence_level=conf_level_str,
                num_sources=len(sources),
                had_warning=is_hallucination if current_mode == "explainable" else False,
                claim_checking_enabled=enable_claim_checking,
                policy_aware_enabled=enable_policy_aware,
                topic_category=topic_cat,
                risk_level=risk_lvl,
                had_policy_disclaimer=had_disclaimer,
                widget_id=f"answer_{latest_message_id}"
            )
            
            # Render accuracy evaluation (for testers)
            render_accuracy_evaluation_widget(
                logger=st.session_state.eval_logger,
                question=prompt,
                answer=answer
            )

        except Exception as e:
            answer = f"⚠️ Error during answer generation:\n\n```\n{traceback.format_exc()}\n```"
            sources = []
            confidence_data = {'level': 'Error', 'emoji': '❌', 'color': '#dc3545', 'avg_distance': 1.0}
            conf_level_str = "Error"
            is_hallucination = True
            policy_modulation = None

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "confidence": conf_level_str if mode == "Explainable AI" else None,
        "had_warning": is_hallucination if mode == "Explainable AI" else None,
        "policy_modulation": policy_modulation if mode == "Explainable AI" else None,
        "question": prompt,
        "mode": current_mode,
        "num_sources": len(sources),
        "claim_checking_enabled": enable_claim_checking,
        "policy_aware_enabled": enable_policy_aware,
        "topic_category": topic_cat,
        "risk_level": risk_lvl,
        "had_policy_disclaimer": had_disclaimer,
        "message_id": latest_message_id
    })
