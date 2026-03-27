# 💌 Wedding Card AI — Description Generator

An end-to-end RAG (Retrieval-Augmented Generation) application that analyzes wedding card images and generates professional, catalog-ready descriptions using Groq LLM, LangChain, and FAISS vector search.

---

## 🗂️ Folder Structure

```
wedding-card-ai/
├── app.py                 # Streamlit frontend — main entry point
├── image_processor.py     # Groq vision: extract visual features from image
├── rag_pipeline.py        # LangChain RAG: prompt engineering + LLM generation
├── vector_store.py        # FAISS vector DB: embeddings + similarity search
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variable template
├── .env                   # Your actual keys (create this, never commit)
├── .cache/                # Auto-created: cached FAISS index
└── README.md              # This file
```

---

## ⚙️ Setup Instructions

### 1. Prerequisites
- Python 3.9 or higher
- A free [Groq API key](https://console.groq.com/)

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
```bash
cp .env.example .env
```
Open `.env` and add your Groq API key:
```
GROQ_API_KEY=gsk_your_actual_key_here
```

### 5. Prepare your JSON dataset
Your JSON file should be a list of objects with this structure:
```json
[
  {
    "SKU": "WC-001",
    "Description": "This premium double-fold wedding card features...",
    "Height": "8 inches",
    "Width": "5.5 inches",
    "Weight": "120g",
    "Image URL": "https://example.com/card.jpg"
  }
]
```

---

## 🚀 Running the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

### Using the app:
1. **Upload your JSON dataset** in the sidebar
2. **Select a style preference** (Auto / Affordable / Premium)
3. **Upload a wedding card image** (JPG, PNG, or WEBP)
4. **Click "Generate Description"**
5. View the generated description + similar cards retrieved

---

## 🧠 How It Works

```
[Wedding Card Image]
        │
        ▼
[Groq Vision Model]           ← llama-4-scout-17b (vision)
  Extracts: colors, theme,
  elements, motifs, finish
        │
        ▼
[Natural Language Query]
        │
        ▼
[FAISS Vector Search]         ← sentence-transformers/all-MiniLM-L6-v2
  Retrieves top-k similar
  catalog descriptions
        │
        ▼
[LangChain RAG Prompt]
  Features + Examples
        │
        ▼
[Groq LLM]                    ← llama-3.3-70b-versatile
  Generates structured
  catalog description
        │
        ▼
[Streamlit UI Output]
```

---

## 🔧 Models Used

| Task | Model | Provider |
|------|-------|----------|
| Image feature extraction | `meta-llama/llama-4-scout-17b-16e-instruct` | Groq |
| Text embedding | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace (local) |
| Description generation | `llama-3.3-70b-versatile` | Groq |

---

## ✨ Features

- **RAG-powered**: Uses real examples from your dataset to maintain tone consistency
- **Embedding cache**: FAISS index is cached to `.cache/` — rebuilt only when dataset changes
- **Style control**: Choose affordable/premium tone from the sidebar
- **Similarity scores**: See how closely retrieved examples match the uploaded card
- **Beginner-friendly**: Clean, modular code with comments throughout

---

## 🐛 Troubleshooting

**`GROQ_API_KEY not found`**
→ Make sure your `.env` file exists and contains the key.

**`No valid documents found in the dataset`**
→ Check that your JSON has a `Description` field in each record.

**Vision model error**
→ Ensure your Groq account has access to `llama-4-scout-17b`. Check [Groq's model list](https://console.groq.com/docs/models).

**Slow first run**
→ The first run downloads the sentence-transformer model (~90MB). Subsequent runs use the cache.

---

## 📦 Dependencies Overview

| Package | Purpose |
|---------|---------|
| `streamlit` | Web UI |
| `langchain` + `langchain-community` + `langchain-groq` | LangChain orchestration |
| `groq` | Groq API client |
| `faiss-cpu` | Vector similarity search |
| `sentence-transformers` | Local text embeddings |
| `python-dotenv` | .env file loading |
| `Pillow` | Image processing |
