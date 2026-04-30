"""
PaperSense – utils.py
Hybrid classification (TF-IDF + keyword signals), extractive summarisation,
and keyword extraction. Zero external API dependencies.
"""

import re
import math
import pickle
import numpy as np
from collections import Counter


# ── Text Cleaning ──────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def sentence_tokenize(text: str) -> list:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.split()) > 4]


def word_tokenize(text: str) -> list:
    return re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())


# ── Stop-words ─────────────────────────────────────────────────────────────────

STOPWORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with","by",
    "from","is","are","was","were","be","been","being","have","has","had","do",
    "does","did","will","would","could","should","may","might","shall","this",
    "that","these","those","it","its","we","our","us","they","their","he","she",
    "you","your","i","my","me","as","if","then","than","so","yet","both","each",
    "more","most","other","some","such","no","not","only","same","also","into",
    "through","during","before","after","above","below","between","among","while",
    "paper","propose","present","method","approach","model","show","result","use",
    "using","used","based","new","work","study","find","two","one","three","four",
    "five","can","via","well","further","first","second","without","however",
    "thus","hence","therefore","whereas","which","who","how","when","where","what",
    "data","dataset","experiments","proposed","existing","previous","recent",
    "demonstrate","demonstrates","demonstrated","achieve","achieves","achieved",
    "outperform","outperforms","improve","improves","improved","significantly",
    "state","art","benchmark","performance","accuracy","results","evaluation",
    "training","testing","large","high","low","set","given","within","across",
    "compared","comparison","per","several","many","all","any","over","under",
    "here","there","following","various","general","specific","current","novel",
    "effective","efficient","problem","task","different","multiple","number",
    "applied","apply","shown","show","shows","able","ability","provide","provides",
}


# ── Domain Keyword Signals (for hybrid classifier) ─────────────────────────────

DOMAIN_SIGNALS = {
    "AI / Machine Learning": [
        "reinforcement learning", "deep learning", "neural network", "transformer",
        "language model", "generative", "few-shot", "meta-learning", "gradient",
        "backpropagation", "attention mechanism", "bert", "gpt", "llm", "fine-tuning",
        "self-supervised", "contrastive learning", "variational autoencoder", "diffusion",
        "policy gradient", "reward model", "pretraining", "knowledge distillation",
        "federated learning", "continual learning", "neural architecture",
    ],
    "Computer Vision": [
        "object detection", "semantic segmentation", "image classification",
        "point cloud", "3d reconstruction", "optical flow", "depth estimation",
        "instance segmentation", "video understanding", "visual recognition",
        "pixel", "bounding box", "convolutional", "mAP", "panoptic segmentation",
        "stereo matching", "camera", "lidar", "image synthesis", "super resolution",
        "face recognition", "image inpainting", "pose estimation",
    ],
    "Data Science / Analytics": [
        "churn", "anomaly detection", "time series", "ETL", "data pipeline",
        "A/B testing", "survival analysis", "recommendation system", "feature engineering",
        "demand forecasting", "fraud detection", "SHAP", "explainability",
        "dashboard", "data warehouse", "clickstream", "tabular data", "imputation",
        "business intelligence", "analytics", "customer segmentation",
    ],
    "Cybersecurity": [
        "intrusion detection", "malware", "vulnerability", "exploit", "attack",
        "threat", "zero-day", "ransomware", "phishing", "honeypot", "encryption",
        "authentication", "side-channel", "fuzzing", "adversarial attack",
        "zero-knowledge", "cyber", "botnet", "DDoS", "penetration testing",
        "memory forensics", "supply chain security", "CVE",
    ],
    "Systems / Software Engineering": [
        "compiler", "operating system", "microservices", "kubernetes", "container",
        "consensus algorithm", "serverless", "garbage collection", "concurrency",
        "file system", "memory management", "CI/CD", "software testing",
        "static analysis", "formal verification", "kernel", "database query",
        "storage engine", "latency", "throughput", "distributed system",
        "program synthesis", "just-in-time compilation",
    ],
}


# ── Hybrid Classification ──────────────────────────────────────────────────────

def hybrid_predict(text: str, clf, vec):
    """
    Combines TF-IDF + Logistic Regression probability with keyword-signal scoring.
    ML weight 60%, keyword signal weight 40%.
    Returns (predicted_domain, confidence_pct, {domain: prob} dict).
    """
    text_lower = text.lower()

    # ── ML base probabilities ──
    X = vec.transform([text])
    proba = clf.predict_proba(X)[0]
    classes = list(clf.classes_)

    # ── Keyword signal scores ──
    signal_scores = {
        domain: sum(1 for kw in keywords if kw in text_lower)
        for domain, keywords in DOMAIN_SIGNALS.items()
    }
    total_signal = sum(signal_scores.values()) or 1

    # ── Combine ──
    combined = {}
    for i, cls in enumerate(classes):
        ml_score = float(proba[i])
        kw_score = signal_scores.get(cls, 0) / total_signal
        combined[cls] = 0.60 * ml_score + 0.40 * kw_score

    total = sum(combined.values()) or 1
    combined = {k: v / total for k, v in combined.items()}

    predicted  = max(combined, key=combined.get)
    confidence = combined[predicted] * 100
    return predicted, confidence, combined


# ── Extractive Summarisation ───────────────────────────────────────────────────

def summarize(text: str, num_sentences: int = 3) -> str:
    sentences = sentence_tokenize(text)
    if len(sentences) <= num_sentences:
        return " ".join(sentences)

    words = [w for w in word_tokenize(text) if w not in STOPWORDS]
    tf = Counter(words)
    total = sum(tf.values()) or 1

    def score(sent: str) -> float:
        sw = [w for w in word_tokenize(sent) if w not in STOPWORDS]
        if not sw:
            return 0.0
        return sum(tf[w] / total for w in sw) / math.sqrt(len(sw))

    scored = sorted(enumerate(sentences), key=lambda x: score(x[1]), reverse=True)
    top_indices = sorted(i for i, _ in scored[:num_sentences])
    return " ".join(sentences[i] for i in top_indices)


# ── Keyword Extraction ─────────────────────────────────────────────────────────

_COMMON_ML_WORDS = {
    "system","network","learning","model","method","data","result","based",
    "using","approach","proposed","paper","algorithm","training","feature",
    "task","classification","detection","analysis","neural","deep","machine",
    "performance","accuracy","evaluation","benchmark","architecture","dataset",
    "image","text","language","graph","function","process","application",
}

def extract_keywords(text: str, top_n: int = 10) -> list:
    tokens = word_tokenize(text)
    filtered = [w for w in tokens if w not in STOPWORDS and len(w) > 3]

    bigrams = [
        f"{filtered[i]} {filtered[i+1]}"
        for i in range(len(filtered) - 1)
        if filtered[i] not in _COMMON_ML_WORDS
        and filtered[i + 1] not in _COMMON_ML_WORDS
    ]

    counts = Counter(filtered + bigrams)
    scored = {
        term: count * (0.7 if term in _COMMON_ML_WORDS else 1.2)
        for term, count in counts.items()
    }

    seen: set = set()
    keywords: list = []
    for term in sorted(scored, key=scored.get, reverse=True):
        words_in_term = set(term.split())
        if words_in_term & seen:
            continue
        keywords.append(term)
        seen |= words_in_term
        if len(keywords) == top_n:
            break

    return [kw.title() for kw in keywords]


# ── Domain Metadata ────────────────────────────────────────────────────────────

DOMAIN_META = {
    "AI / Machine Learning": {
        "icon": "🤖", "color": "#6366f1", "bg": "#eef2ff",
    },
    "Computer Vision": {
        "icon": "👁️", "color": "#0ea5e9", "bg": "#e0f2fe",
    },
    "Data Science / Analytics": {
        "icon": "📊", "color": "#10b981", "bg": "#d1fae5",
    },
    "Cybersecurity": {
        "icon": "🔐", "color": "#ef4444", "bg": "#fee2e2",
    },
    "Systems / Software Engineering": {
        "icon": "⚙️", "color": "#f59e0b", "bg": "#fef3c7",
    },
}


# ── Sample Abstracts ───────────────────────────────────────────────────────────

SAMPLE_ABSTRACTS = {
    "AI / Machine Learning": (
        "We introduce RAPID, a reinforcement learning framework that combines model-based "
        "planning with policy gradient optimization. RAPID learns a differentiable world "
        "model alongside the policy, enabling imagined rollouts for value estimation. "
        "On the DeepMind Control Suite, RAPID achieves human-level performance using 10× "
        "fewer environment interactions than model-free baselines. We also apply "
        "contrastive self-supervised pretraining to the observation encoder, improving "
        "sample efficiency further. Ablation studies confirm the world model and adaptive "
        "entropy regularization are critical components."
    ),
    "Computer Vision": (
        "We propose MeshFormer, a vision transformer for real-time 3D mesh reconstruction "
        "from monocular RGB video. A patch-based encoder extracts per-frame features "
        "aggregated via temporal attention. A lightweight decoder predicts vertex "
        "displacements on a canonical mesh template. MeshFormer achieves 35 FPS on a "
        "single GPU, outperforming prior implicit surface methods on Human3.6M and 3DPW "
        "benchmarks by 12% MPJPE. Our instance segmentation masks are used to crop "
        "person bounding boxes before encoding, reducing background noise."
    ),
    "Data Science / Analytics": (
        "We present ChurnGuard, a production customer retention prediction system deployed "
        "at a telecom operator. ChurnGuard fuses behavioural clickstream data, billing "
        "history, and network quality metrics using a gradient-boosted ensemble with "
        "SHAP explainability surfaced to retention agents in real time. In a six-month "
        "A/B testing experiment, ChurnGuard reduced voluntary churn by 18% and increased "
        "campaign ROI by 2.4× over the legacy analytics dashboard. The data pipeline "
        "uses Apache Kafka for streaming ETL and a feature warehouse for time-series joins."
    ),
    "Cybersecurity": (
        "We introduce ShieldNet, a zero-day intrusion detection system for industrial "
        "control networks based on variational autoencoder anomaly scoring. ShieldNet "
        "models normal SCADA traffic without attack labels and raises alerts when "
        "reconstruction error exceeds an adaptive threshold. Evaluated on the SWaT "
        "dataset, ShieldNet detects 96.3% of cyber attacks at a false positive rate "
        "below 0.5%. A honeypot integration captures attacker lateral movement behaviour "
        "for threat intelligence enrichment. The model runs on commodity hardware."
    ),
    "Systems / Software Engineering": (
        "We present AutoScale, a Kubernetes autoscaler driven by a temporal fusion "
        "transformer that forecasts per-microservice CPU and memory demand 10 minutes "
        "ahead. AutoScale pre-provisions containers before traffic spikes, eliminating "
        "cold-start latency in our serverless deployment. A distributed consensus "
        "protocol coordinates scaling decisions across availability zones. Deployed "
        "across 200 microservices in a production e-commerce platform, AutoScale reduced "
        "P99 response latency by 41% and cloud infrastructure costs by 28% versus the "
        "default horizontal pod autoscaler. Released as an open-source Helm chart."
    ),
}