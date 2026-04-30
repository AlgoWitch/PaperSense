# PaperSense

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://papersense-fhrtqrn4tkjkuo6wstmzb2.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikitlearn)

A machine learning tool that classifies research paper abstracts into academic domains and extracts key insights — domain prediction, confidence scoring, extractive summarisation, and keyword extraction — entirely offline, with no external APIs.

**Live Demo → [papersense-fhrtqrn4tkjkuo6wstmzb2.streamlit.app](https://papersense-fhrtqrn4tkjkuo6wstmzb2.streamlit.app)**

---

## The Problem

Researchers and students often waste significant time skimming papers that turn out to be irrelevant to their work. With thousands of papers published daily across CS subfields, filtering by title alone is unreliable. PaperSense solves this by analysing the abstract — the most information-dense part of any paper — and surfacing structured insights in under a second.

---

## What It Does

Paste any research paper abstract and PaperSense returns:

- **Domain Classification** — predicts which of 5 CS research domains the paper belongs to
- **Confidence Score** — shows how certain the model is, with a full probability breakdown across all domains
- **Extractive Summary** — surfaces the 2–3 most informative sentences from the abstract
- **Keyword Extraction** — identifies key technical terms and phrases

Supported domains: AI / Machine Learning · Computer Vision · Data Science / Analytics · Cybersecurity · Systems / Software Engineering

---

## How It Works

```
Abstract Text
      │
      ▼
TF-IDF Vectorizer (8,000 features, unigrams + bigrams)
      │
      ▼
Logistic Regression Classifier
      │
      ├──▶ Class probabilities  ──▶  Hybrid scorer (ML 60% + keyword signal 40%)
      │                                      │
      │                               Domain + Confidence
      │
      ├──▶ Extractive Summariser (TF-based sentence scoring)
      │
      └──▶ Keyword Extractor (single-document TF-IDF, bigram preference)
```

The classifier uses a **hybrid scoring approach** — combining the Logistic Regression's probabilistic output with a domain keyword signal layer. This corrects for the vocabulary overlap inherent in CS subfields (where terms like "neural network" and "model" appear across multiple domains) and produces more reliable confidence estimates on real-world abstracts.

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| ML / NLP | scikit-learn, TF-IDF Vectorizer, Logistic Regression |
| Summarisation | Custom extractive algorithm (TF sentence scoring) |
| Keyword Extraction | Single-document TF-IDF with bigram preference |
| Frontend | Streamlit |
| Deployment | Streamlit Community Cloud |

No paid APIs. No cloud ML services. Runs fully offline after setup.

---

## Project Structure

```
papersense/
├── app.py                  # Streamlit frontend
├── train.py                # Model training script
├── generate_data.py        # Dataset generation
├── utils.py                # Hybrid classifier, summariser, keyword extractor
├── requirements.txt        # Dependencies
├── data/
│   └── abstracts.csv       # 150 curated research abstracts (5 domains × 30)
└── model/
    ├── classifier.pkl      # Trained Logistic Regression model
    └── vectorizer.pkl      # Fitted TF-IDF vectorizer
```

---

## Running Locally

```bash
# 1. Clone the repository
git clone https://github.com/shreyasuman/papersense.git
cd papersense

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate dataset and train the model
python3 generate_data.py
python3 train.py

# 5. Launch the app
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Dataset

The training dataset consists of 150 manually curated research abstracts — 30 per domain — written to reflect real academic writing styles across AI/ML, Computer Vision, Data Science, Cybersecurity, and Systems/Software Engineering. The dataset is generated via `generate_data.py` and saved to `data/abstracts.csv`.

---

## Design Decisions

**Why Logistic Regression over a neural model?**
For a dataset of 150 samples across 5 classes, a neural network would severely overfit. Logistic Regression with TF-IDF features is well-suited to small text classification datasets, trains in under 5 seconds, is fully interpretable, and produces calibrated probabilities.

**Why the hybrid scorer?**
CS subfields share significant vocabulary. A pure ML classifier trained on 30 samples per class tends to produce low, uncertain probabilities on real abstracts. The keyword signal layer anchors predictions to domain-specific terminology, producing more meaningful confidence scores without hardcoding rules.

**Why extractive summarisation?**
Abstractive summarisation (generating new sentences) requires large language models. Extractive summarisation — selecting the most informative existing sentences — works reliably with no model, no API, and no latency.

---

## License

MIT License. Free to use, modify, and distribute.
