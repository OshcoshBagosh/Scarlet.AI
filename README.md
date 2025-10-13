**Research Question:** How can an explainable AI chatbot improve Rutgers students' access to accurate and transparent campus resource information compared to a traditional black-box chatbot?

## Data Collection & Testing

This repository contains the data collection pipeline and testing model for Scarlet.AI, designed to evaluate explainable AI systems against traditional black-box chatbots.

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
git clone git@github.com:OshcoshBagosh/Scarlet.AI.git
cd knowledge/docling
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
