## NB 

To run the code the module of the repo cardiology-gen-ai MUST be installed e.g. via
```
uv pip install -e ../cardiology-gen-ai
```
(change `../cardiology-gen-ai` with the relative path to the repo in your machine).

Other dependencies can be installed using `uv pip install .`

# Data Extract Transform Load (ETL) pipeline for medical protocols

Acquired cardiology protocols from [European Society of Cardiology](https://www.escardio.org/Guidelines) and saved on [this](https://drive.google.com/drive/folders/1rgaemZ4Jetyz98ivTw8fpLIndgZ2jczn?usp=sharing) Google Drive folder.

Those pdfs have later been converted using [PyMuPDF4LLM](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/) and PyMuPDF into Markdown format and saved in the **/mddocs** folder. I also modified the original code to better extract and save the images, since the method indicated in the PyMuPDF4LLM documentation didn't work properly. This way, bigger and less images are extracted, better fitting this problem. Moreover, a post-processing procedure is done, eliminating repetitive useless patterns like "[.....]".

## 🔍 Vector Database and Semantic Search

After conversion to markdown, the protocols are processed through a vector database pipeline for semantic search capabilities using **Qdrant** and **LangChain**.

### 🚀 Quick Start

#### Prerequisites
- Docker and Docker Compose
- Python 3.8+
- Git

#### Setup
1. **Clone the repository**
   ```bash
   git clone https://github.com/cardiology-gen-ai/data-etl.git
   cd data-etl
   ```
2. **Setup**
   ```bash
   make setup
   ```

3. **Access the dashboard**
   - Dashboard: http://localhost:6333/dashboard

4. (Optional)
   ```bash
   # to see logs
   make logs
   ```

## 📁 Project Structure

```
cardio-protocols-etl/
├── docker-compose.yml          # Qdrant container configuration
├── setup.sh                    # Automated setup script
├── .gitignore                  # Git ignore rules
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── Makefile                    # Convenient command shortcuts
├── qdrant_storage/             # Vector database files (auto-created, not in Git)
├── qdrant_config/              # Qdrant configuration files
├── src/
│   ├── converter.py            # PDF to Markdown converter
│   ├── splitter.py             # Document chunking for vector database
│   ├── embedder.py             # Create embeddings from markdown
│   └── vectorstore.py          # Vector database operations
├── pdfdocs/                    # Original PDF files from ESC
└── mddocs/                     # Converted markdown files and images
```

## 🔧 Complete Pipeline Usage

### Step 1: Run the entire pipeline
```bash
make run
```
### Step 2: Semantic Search
```bash
# Test search
make search
# Search for protocols by meaning, not just keywords
python3 vectorstore.py "chest pain evaluation"
python3 vectorstore.py "acute myocardial infarction management"
python3 vectorstore.py "cardiac catheterization indications"
```

## 🔍 ETL Pipeline Details

### Extract (PDF Acquisition)
- **Source**: [European Society of Cardiology Guidelines](https://www.escardio.org/Guidelines)
- **Storage**: [Google Drive Shared Folder](https://drive.google.com/drive/folders/1rgaemZ4Jetyz98ivTw8fpLIndgZ2jczn?usp=sharing)
- **Format**: PDF documents

### Transform (PDF → Markdown)
- **Tool**: PyMuPDF4LLM with custom modifications
- **Features**:
  - Enhanced image extraction (optimized size and quality)
  - Post-processing to remove repetitive patterns
  - Structured markdown output with proper headers
  - Page break handling for better document structure

### Load (Vector Database)
- **Database**: Qdrant vector database
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Chunking**: LangChain with markdown-aware splitting
- **Features**: Semantic similarity search, metadata filtering

## 🏥 Semantic Search Capabilities

### Query Examples
The vector database enables intelligent search beyond keyword matching:

```python
from vectorstore import similarity_search

# Find protocols related to emergency cardiac care
results = similarity_search("emergency cardiac intervention", k=5)

# Search for diagnostic procedures
results = similarity_search("echocardiography indications", k=3)

# Look for treatment guidelines
results = similarity_search("anticoagulation therapy protocols", k=5)
```

### Search Features
- **Semantic Understanding**: Finds relevant content even with different terminology
- **Medical Context**: Trained on medical literature for better domain understanding
- **Metadata Filtering**: Search within specific protocols or sections
- **Similarity Scoring**: Ranked results by relevance

## 👥 Team Collaboration

### For New Team Members
1. **Clone the repository** (PDF and vector database files are not included)
2. **Download PDFs** from the [shared Google Drive](https://drive.google.com/drive/folders/1rgaemZ4Jetyz98ivTw8fpLIndgZ2jczn?usp=sharing)
3. **Run the complete pipeline**:
   ```bash
   make run
   ```

### Data Sharing Strategy
The repository excludes large files to keep it manageable:
- ✅ **Included**: Source code, configuration, documentation
- ❌ **Excluded**: PDF files, vector database files, extracted images
- 🔗 **Shared separately**: PDFs via Google Drive, regenerated vector DB

## 📊 Database Collections

| Collection | Description | Documents | Vector Dim | Source |
|------------|-------------|-----------|------------|---------|
| `cardio_protocols` | ESC cardiology guidelines | 27 | 384 | ESC Guidelines |

## 🛠️ Configuration and Customization

### Converter Settings
Modify `converter.py` parameters:
- **Image extraction**: Adjust size thresholds and quality
- **Text processing**: Configure pattern removal
- **Output format**: Customize markdown structure

### Vector Database Settings
Configure in `docker-compose.yml`:
- **Memory limits**: Adjust based on document volume
- **Collection settings**: Modify vector dimensions and distance metrics
- **API access**: Enable authentication for production use

### Embedding Model
Change embedding model in `embedder.py`:
```python
# Current: all-MiniLM-L6-v2 (384 dimensions, fast)
# Alternative: all-mpnet-base-v2 (768 dimensions, more accurate)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
```

## 📈 Performance and Monitoring

### Vector Database Stats
```bash
# Check database health
curl http://localhost:6333/health

# View collection info
curl http://localhost:6333/collections/cardio_protocols

# Monitor container
docker-compose logs -f qdrant
```


## 📚 Resources and References

### Medical Guidelines
- [European Society of Cardiology](https://www.escardio.org/Guidelines)
- [ESC Clinical Practice Guidelines](https://www.escardio.org/Guidelines/Clinical-Practice-Guidelines)

### Technical Documentation
- [PyMuPDF4LLM Documentation](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/)
- [Qdrant Vector Database](https://qdrant.tech/documentation/)
- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [Sentence Transformers](https://www.sbert.net/)

