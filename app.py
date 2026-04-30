"""
PaperSense – app.py
Run:  streamlit run app.py
"""

import pickle
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
    page_title="PaperSense",
    page_icon="assets/favicon.png" if False else None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"], .stApp {
    font-family: 'IBM Plex Sans', sans-serif;
    background: #F7F6F3;
    color: #1a1a1a;
}

#MainMenu, footer, header, section[data-testid="stSidebar"] { display: none !important; }

/* ── Layout wrapper ── */
.block-container {
    max-width: 900px !important;
    padding: 3rem 2rem 4rem !important;
    margin: 0 auto !important;
}

/* ── Top bar ── */
.topbar {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    border-bottom: 1.5px solid #1a1a1a;
    padding-bottom: 1rem;
    margin-bottom: 2.5rem;
}
.topbar-brand {
    font-size: 1.1rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #1a1a1a;
}
.topbar-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #888;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* ── Hero ── */
.hero-block {
    margin-bottom: 3rem;
}
.hero-headline {
    font-size: 2.8rem;
    font-weight: 300;
    line-height: 1.2;
    color: #1a1a1a;
    letter-spacing: -0.02em;
    margin-bottom: 1rem;
}
.hero-headline strong {
    font-weight: 600;
}
.hero-desc {
    font-size: 1rem;
    color: #555;
    line-height: 1.7;
    max-width: 560px;
    font-weight: 300;
}

/* ── How it works strip ── */
.how-strip {
    display: flex;
    gap: 0;
    border: 1.5px solid #1a1a1a;
    border-radius: 4px;
    margin-bottom: 2.5rem;
    overflow: hidden;
}
.how-step {
    flex: 1;
    padding: 1rem 1.2rem;
    border-right: 1px solid #d0cfc9;
}
.how-step:last-child { border-right: none; }
.how-num {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: #aaa;
    letter-spacing: 0.1em;
    margin-bottom: 0.3rem;
}
.how-text {
    font-size: 0.82rem;
    color: #333;
    font-weight: 400;
    line-height: 1.4;
}

/* ── Section label ── */
.sec-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #aaa;
    margin-bottom: 0.6rem;
}

/* ── Sample buttons row ── */
.sample-label {
    font-size: 0.78rem;
    color: #888;
    margin-bottom: 0.5rem;
}

/* ── Streamlit button overrides ── */
.stButton > button {
    background: #fff !important;
    color: #1a1a1a !important;
    border: 1.5px solid #ccc !important;
    border-radius: 3px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    padding: 0.4rem 0.9rem !important;
    transition: border-color 0.15s, background 0.15s !important;
    letter-spacing: 0.01em !important;
}
.stButton > button:hover {
    border-color: #1a1a1a !important;
    background: #f0efe9 !important;
}

/* ── Primary analyze button ── */
div[data-testid="column"]:first-child .stButton > button {
    background: #1a1a1a !important;
    color: #fff !important;
    border-color: #1a1a1a !important;
    font-size: 0.85rem !important;
    padding: 0.55rem 1.5rem !important;
}
div[data-testid="column"]:first-child .stButton > button:hover {
    background: #333 !important;
}

/* ── Text area ── */
textarea {
    background: #fff !important;
    color: #1a1a1a !important;
    border: 1.5px solid #ccc !important;
    border-radius: 3px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 0.9rem !important;
    line-height: 1.65 !important;
    padding: 0.9rem 1rem !important;
    transition: border-color 0.15s !important;
}
textarea:focus {
    border-color: #1a1a1a !important;
    box-shadow: none !important;
}

/* ── Divider ── */
.divider {
    border: none;
    border-top: 1px solid #d0cfc9;
    margin: 2rem 0;
}

/* ── Result section ── */
.result-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1px;
    background: #d0cfc9;
    border: 1px solid #d0cfc9;
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 1.5rem;
}
.result-cell {
    background: #fff;
    padding: 1.4rem 1.6rem;
}
.result-cell-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #aaa;
    margin-bottom: 0.5rem;
}
.result-domain {
    font-size: 1.3rem;
    font-weight: 600;
    color: #1a1a1a;
    margin-bottom: 0.4rem;
}
.result-domain-sub {
    font-size: 0.78rem;
    color: #888;
    font-weight: 300;
}
.result-conf-num {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.6rem;
    font-weight: 500;
    color: #1a1a1a;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.result-conf-label {
    font-size: 0.78rem;
    color: #888;
    font-weight: 300;
}

/* ── Bar chart ── */
.prob-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 0.6rem;
}
.prob-label {
    font-size: 0.78rem;
    color: #555;
    min-width: 200px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.prob-track {
    flex: 1;
    height: 4px;
    background: #e8e7e2;
    border-radius: 2px;
}
.prob-fill {
    height: 4px;
    background: #1a1a1a;
    border-radius: 2px;
}
.prob-pct {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #888;
    min-width: 38px;
    text-align: right;
}

/* ── Summary box ── */
.summary-box {
    background: #fff;
    border: 1px solid #d0cfc9;
    border-radius: 4px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.5rem;
}
.summary-text {
    font-size: 0.92rem;
    color: #333;
    line-height: 1.75;
    font-weight: 300;
}

/* ── Keywords ── */
.kw-box {
    background: #fff;
    border: 1px solid #d0cfc9;
    border-radius: 4px;
    padding: 1.2rem 1.6rem;
    margin-bottom: 1.5rem;
}
.kw-pill {
    display: inline-block;
    background: #F7F6F3;
    border: 1px solid #d0cfc9;
    color: #333;
    border-radius: 2px;
    padding: 0.2rem 0.6rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    margin: 0.15rem;
}

/* ── Footer ── */
.footer {
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid #d0cfc9;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.footer-left {
    font-size: 0.72rem;
    color: #aaa;
    font-family: 'IBM Plex Mono', monospace;
}
.footer-right {
    font-size: 0.72rem;
    color: #aaa;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #1a1a1a !important; }

/* ── Warning ── */
.stAlert {
    border-radius: 3px !important;
    font-size: 0.85rem !important;
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

# ── Top Bar ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="topbar">
    <span class="topbar-brand">PaperSense</span>
    <span class="topbar-tag">Research Paper Classifier &nbsp;·&nbsp; v1.0</span>
</div>
""", unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-block">
    <div class="hero-headline">
        Understand any research paper<br><strong>in seconds.</strong>
    </div>
    <p class="hero-desc">
        PaperSense analyses the abstract of a research paper and tells you which
        academic domain it belongs to, how confident the model is, a concise
        summary of the key findings, and the most important technical terms —
        so you can decide if a paper is worth reading before you open it.
    </p>
</div>
""", unsafe_allow_html=True)

# ── How it works ───────────────────────────────────────────────────────────────

st.markdown("""
<div class="how-strip">
    <div class="how-step">
        <div class="how-num">01</div>
        <div class="how-text">Find a research paper on ArXiv, Google Scholar, or any journal</div>
    </div>
    <div class="how-step">
        <div class="how-num">02</div>
        <div class="how-text">Copy the abstract — the short paragraph at the top of the paper</div>
    </div>
    <div class="how-step">
        <div class="how-num">03</div>
        <div class="how-text">Paste it below and click Analyse</div>
    </div>
    <div class="how-step">
        <div class="how-num">04</div>
        <div class="how-text">Get domain, confidence, summary, and keywords instantly</div>
    </div>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error("Model not found. Run `python3 train.py` first.")
    st.stop()

# ── Sample Abstracts ───────────────────────────────────────────────────────────

st.markdown('<p class="sample-label">No abstract handy? Try a sample:</p>', unsafe_allow_html=True)

domain_list = list(SAMPLE_ABSTRACTS.items())
cols = st.columns(len(domain_list))
DOMAIN_SHORT = {
    "AI / Machine Learning": "AI / ML",
    "Computer Vision": "Computer Vision",
    "Data Science / Analytics": "Data Science",
    "Cybersecurity": "Cybersecurity",
    "Systems / Software Engineering": "Systems / SE",
}
for idx, (domain, abstract) in enumerate(domain_list):
    with cols[idx]:
        if st.button(DOMAIN_SHORT[domain], key=f"sample_{idx}", use_container_width=True):
            st.session_state["abstract_text"] = abstract
            st.rerun()

st.markdown("<div style='margin-bottom:1rem'></div>", unsafe_allow_html=True)

# ── Input ──────────────────────────────────────────────────────────────────────

st.markdown('<div class="sec-label">Abstract Input</div>', unsafe_allow_html=True)

default_text = st.session_state.get("abstract_text", "")

abstract = st.text_area(
    label="",
    value=default_text,
    height=180,
    placeholder="Paste the abstract of a research paper here. Typically found at the top of any academic paper, abstracts are 150–300 words and summarise the paper's purpose, methods, and findings.",
)

col_a, col_b, col_c = st.columns([2, 1.2, 6])
with col_a:
    analyze = st.button("Analyse Abstract", use_container_width=True)
with col_b:
    if st.button("Clear", key="clear"):
        st.session_state["abstract_text"] = ""
        st.rerun()

# ── Results ────────────────────────────────────────────────────────────────────

if analyze:
    text = clean_text(abstract)
    if len(text.split()) < 20:
        st.warning("Please paste a longer abstract — at least 20 words are needed for a reliable classification.")
        st.stop()

    with st.spinner("Analysing…"):
        domain, confidence, prob_dict = hybrid_predict(text, classifier, vectorizer)
        summary  = summarize(text, num_sentences=3)
        keywords = extract_keywords(text, top_n=10)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Results</div>', unsafe_allow_html=True)

    # ── Domain + Confidence ────────────────────────────────────────────────────
    conf_label = (
        "High confidence" if confidence >= 70
        else "Moderate confidence" if confidence >= 50
        else "Low confidence — paper may span multiple domains"
    )

    st.markdown(f"""
    <div class="result-grid">
        <div class="result-cell">
            <div class="result-cell-label">Predicted Domain</div>
            <div class="result-domain">{domain}</div>
            <div class="result-domain-sub">Best match out of 5 research domains</div>
        </div>
        <div class="result-cell">
            <div class="result-cell-label">Confidence Score</div>
            <div class="result-conf-num">{confidence:.1f}%</div>
            <div class="result-conf-label">{conf_label}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Probability Breakdown ──────────────────────────────────────────────────
    st.markdown('<div class="sec-label" style="margin-bottom:0.8rem">Domain Probability Breakdown</div>', unsafe_allow_html=True)

    sorted_pairs = sorted(prob_dict.items(), key=lambda x: -x[1])
    for cls, prob in sorted_pairs:
        pct = prob * 100
        is_top = cls == domain
        label_style = "font-weight:500; color:#1a1a1a;" if is_top else ""
        fill_color = "#1a1a1a" if is_top else "#ccc"
        st.markdown(f"""
        <div class="prob-row">
            <div class="prob-label" style="{label_style}">{cls}</div>
            <div class="prob-track">
                <div class="prob-fill" style="width:{pct:.1f}%; background:{fill_color}"></div>
            </div>
            <div class="prob-pct">{pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom:1.5rem'></div>", unsafe_allow_html=True)

    # ── Summary ────────────────────────────────────────────────────────────────
    st.markdown('<div class="sec-label">Summary</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="summary-box">
        <p class="summary-text">{summary}</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Keywords ───────────────────────────────────────────────────────────────
    st.markdown('<div class="sec-label">Key Technical Terms</div>', unsafe_allow_html=True)
    kw_pills = "".join(f'<span class="kw-pill">{kw}</span>' for kw in keywords)
    st.markdown(f"""
    <div class="kw-box">
        {kw_pills}
    </div>
    """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="footer">
    <span class="footer-left">TF-IDF · Logistic Regression · Extractive NLP · No external APIs</span>
    <span class="footer-right">PaperSense · 2025</span>
</div>
""", unsafe_allow_html=True)