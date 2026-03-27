"""
app.py
------
Streamlit frontend for the Wedding Card AI Description Generator.
Orchestrates image upload → feature extraction → RAG retrieval → description generation.
"""

import os
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

from image_processor import extract_image_features, features_to_query_string
from vector_store import build_vector_store, retrieve_similar_cards
from rag_pipeline import generate_description

load_dotenv()

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wedding Card AI · Description Generator",
    page_icon="💌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400&family=DM+Sans:wght@300;400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
  }

  h1, h2, h3 {
    font-family: 'Cormorant Garamond', serif !important;
  }

  .hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3rem;
    font-weight: 600;
    color: #1a0a00;
    line-height: 1.1;
    margin-bottom: 0.25rem;
  }

  .hero-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-weight: 300;
    color: #7a5c3a;
    font-size: 1.05rem;
    letter-spacing: 0.03em;
  }

  .card-description-box {
    background: linear-gradient(135deg, #fdf6ee 0%, #fff8f0 100%);
    border-left: 4px solid #c9893a;
    border-radius: 0 12px 12px 0;
    padding: 1.5rem 2rem;
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    line-height: 1.8;
    color: #2d1a00;
    box-shadow: 0 4px 20px rgba(201,137,58,0.08);
    white-space: pre-line;
  }

  .feature-pill {
    display: inline-block;
    background: #f5e6d0;
    color: #7a4a10;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.8rem;
    margin: 3px 3px;
    font-weight: 500;
  }

  .section-header {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.4rem;
    color: #1a0a00;
    border-bottom: 1px solid #e8d5b5;
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
  }

  .stButton > button {
    background: linear-gradient(135deg, #c9893a, #a86820);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    font-size: 1rem;
    padding: 0.65rem 2rem;
    width: 100%;
    transition: all 0.2s ease;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #a86820, #8a5218);
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(168,104,32,0.3);
  }

  .sidebar-note {
    background: #fdf0e0;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.83rem;
    color: #6b4020;
    margin-top: 0.5rem;
  }

  .step-badge {
    display: inline-block;
    background: #c9893a;
    color: white;
    border-radius: 50%;
    width: 24px;
    height: 24px;
    text-align: center;
    line-height: 24px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 8px;
  }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 💌 Wedding Card AI")
    st.markdown("---")

    st.markdown("**Dataset**")
    dataset_file = st.file_uploader(
        "Upload JSON dataset",
        type=["json"],
        help="Upload your wedding card descriptions JSON file",
    )

    st.markdown("---")
    st.markdown("**Style Preference**")
    style_preference = st.radio(
        "Tone of description",
        options=["auto", "affordable", "premium"],
        format_func=lambda x: {
            "auto": "🔍 Auto-detect",
            "affordable": "💰 Affordable",
            "premium": "👑 Premium",
        }[x],
        index=0,
    )

    st.markdown("---")
    st.markdown("""
    <div class="sidebar-note">
    <strong>How it works:</strong><br>
    <span class="step-badge">1</span> Upload a card image<br>
    <span class="step-badge">2</span> AI extracts visual features<br>
    <span class="step-badge">3</span> RAG retrieves similar cards<br>
    <span class="step-badge">4</span> LLM generates description
    </div>
    """, unsafe_allow_html=True)


# ─── Main Content ────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 1rem 0 2rem 0;">
  <div class="hero-title">Wedding Card Description Generator</div>
  <div class="hero-subtitle">Upload a card image · Get a catalog-ready description instantly</div>
</div>
""", unsafe_allow_html=True)

# Check for API key
if not os.getenv("GROQ_API_KEY"):
    st.error("⚠️ GROQ_API_KEY not found. Please add it to your `.env` file.")
    st.stop()

# Check for dataset
if not dataset_file:
    st.info("👈 Please upload your wedding card JSON dataset in the sidebar to get started.")
    st.stop()

# ─── Load Vector Store ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Building knowledge base from dataset...")
def get_vector_store(file_content_hash: str, _file_content: bytes):
    import tempfile, json, pathlib
    # Write uploaded bytes to a temp file
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".json", delete=False) as tmp:
        tmp.write(_file_content)
        tmp_path = tmp.name
    vs = build_vector_store(tmp_path)
    return vs


file_bytes = dataset_file.read()
file_hash = str(hash(file_bytes))

try:
    with st.spinner("Loading knowledge base..."):
        vectorstore = get_vector_store(file_hash, file_bytes)
    st.success(f"✅ Knowledge base ready — dataset loaded successfully")
except Exception as e:
    st.error(f"❌ Failed to build knowledge base: {e}")
    st.stop()

# ─── Image Upload ────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Upload Wedding Card Image</div>', unsafe_allow_html=True)

col_upload, col_preview = st.columns([1, 1], gap="large")

with col_upload:
    uploaded_image = st.file_uploader(
        "Choose a wedding card image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload a clear, front-facing image of the wedding card",
    )

    if uploaded_image:
        image = Image.open(uploaded_image)
        generate_btn = st.button("✨ Generate Description", use_container_width=True)

with col_preview:
    if uploaded_image:
        st.image(image, caption="Uploaded Wedding Card", use_column_width=True)

# ─── Generation ──────────────────────────────────────────────────────────────
if uploaded_image and generate_btn:
    st.markdown("---")

    # Step 1: Extract features
    with st.status("🔍 Analyzing image...", expanded=True) as status:
        st.write("Extracting visual features with Groq Vision...")
        try:
            features = extract_image_features(image, style_preference)
            st.write("✅ Visual features extracted")

            # Step 2: Build query + retrieve
            st.write("🗂️ Querying knowledge base for similar cards...")
            query = features_to_query_string(features)
            retrieved = retrieve_similar_cards(vectorstore, query, k=5)
            st.write(f"✅ Retrieved {len(retrieved)} similar card descriptions")

            # Step 3: Generate description
            st.write("✍️ Generating catalog description with LLM...")
            result = generate_description(features, retrieved, style_preference)
            st.write("✅ Description generated!")

            status.update(label="✅ Complete!", state="complete", expanded=False)
        except Exception as e:
            status.update(label="❌ Error", state="error")
            st.error(f"Generation failed: {e}")
            st.stop()

    # ─── Results ──────────────────────────────────────────────────────────
    st.markdown("---")
    res_col1, res_col2 = st.columns([3, 2], gap="large")

    with res_col1:
        st.markdown('<div class="section-header">Generated Description</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="card-description-box">{result["description"]}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            label="📋 Copy / Download Description",
            data=result["description"],
            file_name="wedding_card_description.txt",
            mime="text/plain",
        )

    with res_col2:
        st.markdown('<div class="section-header">Visual Features Detected</div>', unsafe_allow_html=True)

        # Show feature pills
        all_tags = (
            features.get("colors", [])
            + features.get("elements", [])
            + features.get("motifs", [])
            + [features.get("theme", ""), features.get("style", ""), features.get("finish", "")]
        )
        all_tags = [t for t in all_tags if t]

        pills_html = "".join(
            f'<span class="feature-pill">{tag}</span>' for tag in all_tags
        )
        st.markdown(pills_html, unsafe_allow_html=True)

        st.markdown(f"""
        <br>
        <table style="font-size:0.88rem; color:#3d2b0e; width:100%;">
          <tr><td><b>Card Type</b></td><td>{features.get('card_type','—')}</td></tr>
          <tr><td><b>Theme</b></td><td>{features.get('theme','—')}</td></tr>
          <tr><td><b>Style</b></td><td>{features.get('style','—')}</td></tr>
          <tr><td><b>Paper</b></td><td>{features.get('paper_quality','—')}</td></tr>
          <tr><td><b>Finish</b></td><td>{features.get('finish','—')}</td></tr>
        </table>
        """, unsafe_allow_html=True)

elif not uploaded_image:
    st.markdown("""
    <div style="text-align:center; padding: 3rem; color: #9a7550; border: 2px dashed #e8d5b5; border-radius: 16px; margin-top: 1rem;">
      <div style="font-size: 3rem; margin-bottom: 1rem;">💌</div>
      <div style="font-family: 'Cormorant Garamond', serif; font-size: 1.4rem; color: #7a5030;">
        Upload a wedding card image to get started
      </div>
      <div style="font-size: 0.9rem; margin-top: 0.5rem;">
        Supports JPG, PNG, WEBP formats
      </div>
    </div>
    """, unsafe_allow_html=True)