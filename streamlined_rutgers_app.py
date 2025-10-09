#!/usr/bin/env python3
"""
Streamlined Rutgers AI Assistant - Direct from URLs to ChromaDB
No intermediate JSONL files needed!
"""

import streamlit as st
import chromadb
import google.generativeai as genai
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from utils.tokenizer import OpenAITokenizerWrapper
from typing import List, Dict

# Configure Gemini
genai.configure(api_key="AIzaSyDFtQysHWUJxNaZIXyiDi7GJj32S0nXKd8")
model = genai.GenerativeModel('gemini-2.5-flash')

# Rutgers URLs - Comprehensive List
RUTGERS_URLS = [
    #Registrar (Calendars, Schedules, Policies)
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
    
    #Financial Aid & Student Accounting
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
    
    #Housing & Dining
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
    
    #Student Handbook & Conduct
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
    
    #Student Life & Involvement
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
    
    #Academics - SAS
    "https://sas.rutgers.edu/",
    "https://sas.rutgers.edu/academics",
    "https://sas.rutgers.edu/academics/majors-minors",
    "https://sas.rutgers.edu/academics/departments-programs-and-centers",
    "https://sasundergrad.rutgers.edu/",
    "https://sas.rutgers.edu/about/sas-offices/detail/office-of-advising-and-academic-services",
    "https://sas.rutgers.edu/academics/sas-divisions",
    "https://rge.sas.rutgers.edu/",
    "https://sas.rutgers.edu/students/meet-our-students",
    
    #Academics - SEBS
    "https://sebs.rutgers.edu/",
    "https://sebs.rutgers.edu/academics",
    "https://sebs.rutgers.edu/academics/advisors",
    "https://sebs.rutgers.edu/graduate-programs",
    "https://sebs.rutgers.edu/research",
    "https://sebs.rutgers.edu/beyond-the-classroom",
    "https://extension.rutgers.edu/",
    
    #Academics - SOE
    "https://soe.rutgers.edu/",
    "https://soe.rutgers.edu/academics",
    "https://soe.rutgers.edu/academics/undergraduate",
    "https://soe.rutgers.edu/academics/graduate",
    "https://soe.rutgers.edu/research/departments",
    "https://soe.rutgers.edu/research",
    "https://soe.rutgers.edu/advising",
    
    #Academics - Mason Gross
    "https://www.masongross.rutgers.edu/",
    "https://www.masongross.rutgers.edu/events/",
    "https://www.masongross.rutgers.edu/degrees-programs",
    "https://www.masongross.rutgers.edu/admissions",
    "https://www.masongross.rutgers.edu/calendar",
    "https://www.masongross.rutgers.edu/faculty",
    
    #Academics - RBS
    "https://www.business.rutgers.edu/",
    "https://www.business.rutgers.edu/undergraduate-new-brunswick",
    "https://www.business.rutgers.edu/mba",
    "https://www.business.rutgers.edu/phd",
    "https://www.business.rutgers.edu/executive-education",
    "https://www.business.rutgers.edu/faculty-research",
    
    #Academics - Bloustein
    "https://bloustein.rutgers.edu/",
    "https://bloustein.rutgers.edu/academics/undergraduate",
    "https://bloustein.rutgers.edu/academics/graduate",
    "https://bloustein.rutgers.edu/research",
    "https://bloustein.rutgers.edu/faculty",
    "https://bloustein.rutgers.edu/upcoming-events/",
    
    #Honors College
    "https://honorscollege.rutgers.edu/",
    "https://honorscollege.rutgers.edu/academics/overview",
    "https://honorscollege.rutgers.edu/academics/academic-affairs-advising",
    "https://honorscollege.rutgers.edu/academics/curriculum",
    "https://honorscollege.rutgers.edu/academics/research",
    "https://honorscollege.rutgers.edu/admissions",
    "https://honorscollege.rutgers.edu/current-students",
    "https://honorscollege.rutgers.edu/admissions/transfer-sophomores-and-juniors",
    
    #Health & Wellness
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
    
    #Title IX & Safety
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
    
    #Transportation
    "https://ipo.rutgers.edu/dots",
    "https://ipo.rutgers.edu/transportation",
    "https://ipo.rutgers.edu/transportation/buses/nb",
    "https://ipo.rutgers.edu/parking",
    "https://rutgers.passiogo.com/"
]

@st.cache_resource
def setup_chromadb():
    """Setup ChromaDB and load data if needed"""
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Check if collection exists and has data
    try:
        collection = client.get_collection("rutgers_docs")
        if collection.count() > 0:
            return collection
    except:
        pass  # Collection doesn't exist, which is fine
    
    # Create collection and load data
    collection = client.create_collection(
        name="rutgers_docs",
        metadata={"description": "Rutgers University documents and information"}
    )
    
    load_rutgers_data_to_chromadb(collection)
    
    return collection

def load_rutgers_data_to_chromadb(collection):
    """Load Rutgers data directly from URLs to ChromaDB"""
    
    # Initialize tokenizer and chunker
    tokenizer = OpenAITokenizerWrapper()
    MAX_TOKENS = 8191
    
    # Convert documents
    converter = DocumentConverter()
    results = converter.convert_all(RUTGERS_URLS, raises_on_error=False)
    
    # Process documents and create URL mapping
    docs = []
    url_mapping = {}  # Map filename to full URL
    
    for i, result in enumerate(results):
        if result.document:
            docs.append(result.document)
            # Store the mapping from filename to full URL
            # Try different ways to get the filename
            filename = f"doc_{i}"
            if hasattr(result.document, 'meta') and hasattr(result.document.meta, 'origin'):
                if hasattr(result.document.meta.origin, 'filename'):
                    filename = result.document.meta.origin.filename
            elif hasattr(result.document, 'origin'):
                if hasattr(result.document.origin, 'filename'):
                    filename = result.document.origin.filename
            elif hasattr(result.document, 'filename'):
                filename = result.document.filename
            
            url_mapping[filename] = RUTGERS_URLS[i] if i < len(RUTGERS_URLS) else f"https://rutgers.edu/{filename}"
    
    # Chunk documents
    chunker = HybridChunker(
        tokenizer=tokenizer,
        max_tokens=MAX_TOKENS,
        merge_peers=True,
    )
    
    all_chunks = []
    for i, doc in enumerate(docs):
        try:
            chunk_iter = chunker.chunk(dl_doc=doc)
            all_chunks.extend(list(chunk_iter))
        except Exception as e:
            st.warning(f"Chunking failed: {e}")
    
    #Prepare data for ChromaDB
    documents = []
    ids = []
    metadatas = []
    
    for i, chunk in enumerate(all_chunks):
        # Extract title and URL. Try to get better title from document metadata
        title = "Rutgers Document"
        filename = f"doc_{i}"
        
        if hasattr(chunk, 'meta'):
            if hasattr(chunk.meta, 'headings') and chunk.meta.headings:
                title = chunk.meta.headings[0]
            elif hasattr(chunk.meta, 'title') and chunk.meta.title:
                title = chunk.meta.title
            
            if hasattr(chunk.meta, 'origin') and hasattr(chunk.meta.origin, 'filename'):
                filename = chunk.meta.origin.filename
        elif hasattr(chunk, 'origin') and hasattr(chunk.origin, 'filename'):
            filename = chunk.origin.filename
        elif hasattr(chunk, 'filename'):
            filename = chunk.filename
        
        # Use the filename as title if no other title available
        if title == "Rutgers Document":
            title = filename.replace('-', ' ').replace('_', ' ').title()
        
        url = url_mapping.get(filename, f"https://rutgers.edu/{filename}")
        
        
        documents.append(chunk.text)
        ids.append(f"rutgers_chunk_{i+1}")
        metadatas.append({
            'title': title,
            'url': url
        })
    
    #Add to ChromaDB
    collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas
    )
    
    st.success(f"✅ Loaded {len(documents)} chunks into ChromaDB")

def search_rutgers_semantic(collection, query: str, n_results: int = 5):
    """Search Rutgers data using semantic search"""
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    #Format results
    formatted_results = []
    for i in range(len(results['documents'][0])):
        formatted_results.append({
            'text': results['documents'][0][i],
            'title': results['metadatas'][0][i]['title'],
            'url': results['metadatas'][0][i]['url'],
            'distance': results['distances'][0][i]
        })
    
    return formatted_results

def ask_rutgers_question(collection, question: str):
    """Ask a question about Rutgers using semantic search + Gemini"""
    
    # Search for relevant chunks using semantic search - get more results for better coverage
    relevant_chunks = search_rutgers_semantic(collection, question, n_results=10)
    
    if not relevant_chunks:
        return "Sorry, I couldn't find relevant information about that topic in the Rutgers data.", []
    
    # Prepare context
    context_parts = []
    for chunk in relevant_chunks:
        context_parts.append(f"Source: {chunk['title']}\nText: {chunk['text']}")
    
    context = "\n\n".join(context_parts)
    
    # Create prompt
    prompt = f"""
    You are a helpful AI assistant for Rutgers University. 
    Answer the following question based *only* on the provided context.
    If the answer is not in the context, say "Sorry, I couldn't find relevant information about that topic in the Rutgers data."
    
    Question: {question}
    
    Context:
    {context}
    """
    
    #Gemini response
    response = model.generate_content(prompt)
    return response.text, relevant_chunks

# ---------------------------------------------- Streamlit UI ----------------------------------------------
# Streamlit UI
st.set_page_config(
    page_title="Rutgers AI Assistant",
    page_icon="🏛️",
    layout="wide"
)

# Streamlit CSS
st.markdown("""
<style>
.main-header {
    color: #CC0000;
    text-align: center;
    font-size: 2.5em;
    margin-bottom: 1em;
}
.source-box {
    background-color: #f9f9f9;
    padding: 0.5rem;
    border-radius: 5px;
    margin: 0.5rem 0;
    border-left: 3px solid #CC0000;
    color: #000000;
}
.source-box a {
    color: #CC0000;
    text-decoration: none;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Setup ChromaDB
collection = setup_chromadb()

# Header
st.markdown('<h1 class="main-header">🏛️ Rutgers University AI Assistant</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📊 Database Info")
    st.write(f"Documents loaded: {collection.count()}")
    
    if st.button("🔄 Reload Data"):
        st.cache_resource.clear()
        st.rerun()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message["role"] == "assistant" and "sources" in message:
            with st.expander(f"📚 Sources ({len(message['sources'])} found)"):
                for i, source in enumerate(message["sources"]):
                    # Use the actual URL from the source
                    url = source['url']
                    
                    st.markdown(f"""
                    <div class="source-box">
                        <strong>{i+1}. {source['title']}</strong>
                        <br><small>Relevance: {source['distance']:.3f} | <a href="{url}" target="_blank">🔗 View Source</a></small>
                    </div>
                    """, unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask me anything about Rutgers University..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    #Get response
    with st.chat_message("assistant"):
        with st.spinner("Searching Rutgers knowledge base..."):
            answer, sources = ask_rutgers_question(collection, prompt)
        
        st.markdown(answer)
        
        if sources:
            with st.expander(f"📚 Sources ({len(sources)} found)"):
                for i, source in enumerate(sources):
                    # Use the actual URL from the source
                    url = source['url']
                    
                    st.markdown(f"""
                    <div class="source-box">
                        <strong>{i+1}. {source['title']}</strong>
                        <br><small>Relevance: {source['distance']:.3f} | <a href="{url}" target="_blank">🔗 View Source</a></small>
                    </div>
                    """, unsafe_allow_html=True)
    
    #Add the responses to chat history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer,
        "sources": sources
    })
