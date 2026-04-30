"""
PaperSense – app.py
Run:  streamlit run app.py
"""

import pickle
import numpy as np
import streamlit as st
from utils import (
    hybrid_predict,
    clean_text,
    summarize,
    extract_keywords,
    DOMAIN_META,
    SAMPLE_ABSTRACTS,
)

# ── Page Config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PaperSense – Research Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0f1117;
    color: #e2e8f0;
}

#MainMenu, footer, header { visibility: hidden; }

.hero {
    background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 50%, #0c1a2e 100%);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(99,102,241,0.18) 0%, transparent 70%);
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a5b4fc, #818cf8, #6ee7b7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem;
    letter-spacing: -0.5px;
}
.hero-sub {
    font-size: 1.05rem;
    color: #94a3b8;
    margin: 0;
    font-weight: 300;
}

.section-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #6366f1;
    margin-bottom: 0.4rem;
}

.result-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}
.result-card:hover { border-color: #6366f1; }

.domain-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.45rem 1rem;
    border-radius: 999px;
    font-size: 0.95rem;
    font-weight: 600;
    margin-bottom: 0.6rem;
}

.conf-number {
    font-family: 'DM Mono', monospace;
    font-size: 2.8rem;
    font-weight: 500;
    line-height: 1;
}

.kw-pill {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.35);
    color: #a5b4fc;
    border-radius: 6px;
    padding: 0.25rem 0.65rem;
    font-size: 0.8rem;
    font-weight: 500;
    margin: 0.2rem;
    font-family: 'DM Mono', monospace;
}

.summary-text {
    color: #cbd5e1;
    line-height: 1.75;
    font-size: 0.97rem;
}

.bar-track {
    background: #334155;
    border-radius: 999px;
    height: 8px;
    width: 100%;
    margin-top: 0.8rem;
}
.bar-fill {
    height: 8px;
    border-radius: 999px;
}

section[data-testid="stSidebar"] {
    background: #0f172a !important;
    border-right: 1px solid #1e293b;
}
section[data-testid="stSidebar"] * {
    color: #94a3b8 !important;
}

textarea {
    background: #0f172a !important;
    color: #e2e8f0 !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
}
textarea:focus { border-color: #6366f1 !important; }

.stButton > button {
    background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.6rem 2rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

hr { border-color: #1e293b !important; }

.stSelectbox > div > div {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load Model ─────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    clf = pickle.load(open("model/classifier.pkl", "rb"))
    vec = pickle.load(open("model/vectorizer.pkl", "rb"))
    return clf, vec

try:
    classifier, vectorizer = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔬 PaperSense")
    st.markdown("---")
    st.markdown(
        "**PaperSense** classifies research paper abstracts into academic domains "
        "and extracts key insights instantly."
    )
    st.markdown("---")
    st.markdown("### 📂 Supported Domains")
    for domain, meta in DOMAIN_META.items():
        st.markdown(f"{meta['icon']} &nbsp; {domain}", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### ⚙️ Model Info")
    st.markdown("- **Classifier**: Logistic Regression")
    st.markdown("- **Features**: TF-IDF (1-2 grams, 8k vocab)")
    st.markdown("- **Summary**: Extractive (TF scoring)")
    st.markdown("- **Keywords**: Single-doc TF-IDF")
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.78rem;color:#475569'>"
        "Built with Python · scikit-learn · Streamlit<br>"
        "© 2025 PaperSense"
        "</div>",
        unsafe_allow_html=True,
    )

# ── Hero ───────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
  <div class="hero-title">🔬 PaperSense</div>
  <p class="hero-sub">Paste a research abstract → Instantly get domain, confidence, summary & keywords</p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(
        "⚠️ Model files not found. Please run `python3 train.py` first to generate "
        "`model/classifier.pkl` and `model/vectorizer.pkl`."
    )
    st.stop()

# ── Sample Abstract Buttons ────────────────────────────────────────────────────

st.markdown('<div class="section-label">Try a Sample Abstract</div>', unsafe_allow_html=True)

sample_cols = st.columns(len(SAMPLE_ABSTRACTS))
for idx, (domain, abstract) in enumerate(SAMPLE_ABSTRACTS.items()):
    meta = DOMAIN_META[domain]
    with sample_cols[idx]:
        if st.button(f"{meta['icon']} {domain.split('/')[0].strip()}", key=f"sample_{idx}"):
            st.session_state["abstract_text"] = abstract
            st.rerun()

# ── Text Input ─────────────────────────────────────────────────────────────────

default_text = st.session_state.get("abstract_text", "")

st.markdown('<div class="section-label" style="margin-top:1.5rem">Abstract Input</div>', unsafe_allow_html=True)
abstract = st.text_area(
    label="",
    value=default_text,
    height=200,
    placeholder="Paste your research paper abstract here…",
   
)

col_btn, col_clear = st.columns([2, 8])
with col_btn:
    analyze = st.button("🔍 Analyze", use_container_width=True)
with col_clear:
    if st.button("✕ Clear", key="clear"):
        st.session_state["abstract_text"] = ""
        st.rerun()

# ── Analysis ───────────────────────────────────────────────────────────────────

if analyze:
    text = clean_text(abstract)
    if len(text.split()) < 20:
        st.warning("⚠️ Please enter at least 20 words for a meaningful analysis.")
        st.stop()

    with st.spinner("Analysing abstract…"):
        domain, confidence, prob_dict = hybrid_predict(text, classifier, vectorizer)
        summary  = summarize(text, num_sentences=3)
        keywords = extract_keywords(text, top_n=10)
        meta     = DOMAIN_META[domain]

    st.markdown("---")
    st.markdown('<div class="section-label">Analysis Results</div>', unsafe_allow_html=True)

    # ── Domain + Confidence ────────────────────────────────────────────────────
    col_domain, col_conf = st.columns([1, 1])

    with col_domain:
        bar_pct = int(confidence)
        st.markdown(f"""
        <div class="result-card">
          <div class="section-label">Predicted Domain</div>
          <div class="domain-badge" style="background:{meta['bg']}; color:{meta['color']}">
            {meta['icon']} &nbsp; {domain}
          </div>
          <div class="bar-track">
            <div class="bar-fill" style="width:{bar_pct}%; background:{meta['color']}"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_conf:
        conf_color = (
            "#10b981" if confidence >= 70
            else "#f59e0b" if confidence >= 50
            else "#ef4444"
        )
        conf_label = (
            "High Confidence" if confidence >= 70
            else "Moderate Confidence" if confidence >= 50
            else "Low Confidence"
        )
        st.markdown(f"""
        <div class="result-card">
          <div class="section-label">Confidence Score</div>
          <div class="conf-number" style="color:{conf_color}">{confidence:.1f}%</div>
          <div style="font-size:0.82rem; color:#64748b; margin-top:0.3rem">{conf_label}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Probability Breakdown ──────────────────────────────────────────────────
    st.markdown('<div class="section-label" style="margin-top:0.5rem">Domain Probability Breakdown</div>', unsafe_allow_html=True)

    sorted_pairs = sorted(prob_dict.items(), key=lambda x: -x[1])
    for cls, prob in sorted_pairs:
        m   = DOMAIN_META[cls]
        pct = prob * 100
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:1rem; margin-bottom:0.5rem">
          <div style="min-width:220px; font-size:0.87rem; color:#94a3b8">
            {m['icon']} {cls}
          </div>
          <div style="flex:1; background:#1e293b; border-radius:999px; height:6px">
            <div style="width:{pct:.1f}%; background:{m['color']}; height:6px; border-radius:999px"></div>
          </div>
          <div style="min-width:48px; text-align:right; font-family:'DM Mono',monospace;
                      font-size:0.82rem; color:{m['color']}">{pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Summary ────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="result-card" style="margin-top:1rem">
      <div class="section-label">AI Summary</div>
      <p class="summary-text">{summary}</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Keywords ───────────────────────────────────────────────────────────────
    kw_pills = "".join(f'<span class="kw-pill">{kw}</span>' for kw in keywords)
    st.markdown(f"""
    <div class="result-card">
      <div class="section-label">Key Technical Terms</div>
      <div style="margin-top:0.4rem">{kw_pills}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; color:#475569; font-size:0.78rem; margin-top:1.5rem">
      PaperSense uses TF-IDF + Logistic Regression for classification
      and extractive algorithms for summarisation — no external APIs required.
    </div>
    """, unsafe_allow_html=True)