from __future__ import annotations

import random
import re
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "chatbot_dataset.csv"


def normalize_text(text: str) -> str:
    """Lowercase and keep words plus simple punctuation spacing."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s?!.]", " ", text)
    return re.sub(r"\s+", " ", text)


def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    data = pd.read_csv(DATASET_PATH)
    required_columns = {"Question", "Answer"}
    if not required_columns.issubset(data.columns):
        raise ValueError("chatbot_dataset.csv must contain Question and Answer columns")

    data = data.dropna(subset=["Question", "Answer"]).copy()
    data["Question"] = data["Question"].astype(str).map(normalize_text)
    data["Answer"] = data["Answer"].astype(str).str.strip()
    return data


def build_retriever(data: pd.DataFrame) -> tuple[TfidfVectorizer, object]:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    question_matrix = vectorizer.fit_transform(data["Question"])
    return vectorizer, question_matrix


def estimate_accuracy(data: pd.DataFrame) -> float:
    if len(data) < 4:
        return 1.0

    x_train, x_test, y_train, y_test = train_test_split(
        data["Question"],
        data["Answer"],
        test_size=0.25,
        random_state=42,
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    train_matrix = vectorizer.fit_transform(x_train)

    correct = 0
    for question, expected_answer in zip(x_test, y_test):
        similarity_scores = cosine_similarity(vectorizer.transform([question]), train_matrix)[0]
        best_match_index = int(similarity_scores.argmax())
        predicted_answer = y_train.iloc[best_match_index]
        if predicted_answer == expected_answer:
            correct += 1

    return correct / len(x_test)


def create_fallbacks() -> dict[str, list[str]]:
    return {
        "greeting": [
            "Hello! I can help with study, tech, and project questions.",
            "Hi there. Ask me anything about the project or basic concepts.",
        ],
        "fallback": [
            "I’m not fully sure yet. Try rephrasing your question.",
            "I couldn’t match that confidently. Please ask in a different way.",
        ],
    }


DATA = load_dataset()
VECTORIZER, QUESTION_MATRIX = build_retriever(DATA)
MODEL_ACCURACY = estimate_accuracy(DATA)
FALLBACKS = create_fallbacks()
CONFIDENCE_THRESHOLD = 0.26

app = Flask(__name__)


@app.get("/")
def home():
    return render_template(
        "index.html",
        model_accuracy=f"{MODEL_ACCURACY:.2f}",
        dataset_size=len(DATA),
    )


@app.post("/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    user_message = str(payload.get("message", "")).strip()

    if not user_message:
        return jsonify({"reply": "Please type a message first."})

    normalized_message = normalize_text(user_message)
    similarity_scores = cosine_similarity(
        VECTORIZER.transform([normalized_message]),
        QUESTION_MATRIX,
    )[0]
    best_index = int(similarity_scores.argmax())
    confidence = float(similarity_scores[best_index])
    reply = DATA.iloc[best_index]["Answer"]

    if confidence < CONFIDENCE_THRESHOLD:
        reply = random.choice(FALLBACKS["fallback"])

    return jsonify(
        {
            "reply": reply,
            "confidence": round(confidence, 3),
            "threshold": CONFIDENCE_THRESHOLD,
        }
    )


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True)
