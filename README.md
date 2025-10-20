# Scarlet.AI
An Explainable AI Chatbot for College Students, focused on providing reliable information about Rutgers University.  

## Data Collection & Testing

Scarlet.AI aims to solve this by:  
- Answering questions about Rutgers resources, deadlines, and campus life.  
- Providing source transparency (citations, URLs, snippets).  
- Displaying confidence levels so users know reliability.  
- Flagging possible hallucinations and encouraging double-checks.  
- Comparing a black box chatbot with our explainable version to show the added value.  

Scarlet.AI provides source transparency (citations, URLs) and displays confidence levels so users know reliability. We use Rutgers University's public information as our dataset to test this approach.

**Dataset:** Rutgers-specific sources including official websites, creating a collection of information the chatbot can search and retrieve from when answering student questions.

## Data Collection Process

- **Document Processing**: Docling for web page extraction and chunking
- **Chunking**: Text segmentation for retrieval of source URLs, title and document structure
- **Vector Database**: ChromaDB for semantic search
- **AI Model**: Google Gemini 2.5 Flash
- **Web Interface**: Streamlit

## How to use

### 1. Clone and Setup
```bash
git clone -b data-processing-princess https://github.com/OshcoshBagosh/Scarlet.AI.git
cd Scarlet.AI
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the App
```bash
python3 -m streamlit run streamlined_rutgers_app.py --server.port 8502
```

### 4. Open in Browser
Go to: `http://localhost:8502`

## Git Commands

### Basic Git Workflow
```bash
# Check the status of your repo
git status

# Add files to staging
git add <filename>
git add .   # add all changes

# Commit changes
git commit -m "your commit message"

# Push to GitHub
git push origin main

# Pull the latest changes
git pull origin main
```

### Working with Branches
```bash
# Create a new branch
git checkout -b your-branch-name

# Switch to main branch
git checkout main

# Merge your branch into main
git merge your-branch-name
```
