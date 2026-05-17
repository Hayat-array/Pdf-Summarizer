"""Smart PDF Summarizer

README
======
This Flask app powers a college project that uploads PDFs, extracts text,
generates simple summaries, supports keyword and flashcard generation, and
answers questions with a lightweight sentence-retrieval chatbot.

Main features:
- MongoDB-backed users, PDFs, and summaries
- Flask-Login based authentication
- PDF upload with background extraction using threading
- NLTK-based summarization and keyword extraction
- TF-IDF chatbot for PDF Q&A
- Demo mode when MongoDB is unavailable, so the UI still works
"""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import nltk
import pdfplumber
from bson import ObjectId
from dotenv import load_dotenv
from flask import (
    Flask,
    Response,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from flask_login import (
    LoginManager,
    UserMixin,
    current_user,
    login_required,
    login_user,
    logout_user,
)
from flask_pymongo import PyMongo
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from config import Config


# Always load the project-local .env, regardless of current working directory.
load_dotenv(Path(__file__).resolve().parent / ".env")

app = Flask(__name__)
app.config.from_object(Config)
mongo = PyMongo(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message_category = "warning"

BASE_DIR = Path(app.config["UPLOAD_FOLDER"])
BASE_DIR.mkdir(parents=True, exist_ok=True)

DEMO_MODE = False
DEMO_DB: Dict[str, Dict[str, Dict[str, Any]]] = {
    "users": {},
    "pdfs": {},
    "summaries": {},
}


def _ensure_nltk_data_disabled() -> None:
    """NLTK data validation disabled due to Anaconda corruption issues."""
    pass


# Mongo connectivity is checked after `test_mongo_connection` is defined.


# Fallback tokenizers and stopword set to avoid NLTK data lookup failures.
# These override NLTK tokenizers at runtime so the app remains functional
# even when NLTK corpora (punkt/stopwords) are missing or corrupted.
def _simple_sent_tokenize(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _simple_word_tokenize(text: str) -> List[str]:
    return re.findall(r"\b[a-zA-Z]+\b", text)


# Override imported NLTK tokenizers with safe fallbacks.
sent_tokenize = _simple_sent_tokenize
word_tokenize = _simple_word_tokenize


# Safe stopword loader: try NLTK, fall back to a small built-in set.
try:
    _nltk_stopwords = set(stopwords.words("english"))
except Exception:
    _nltk_stopwords = {
        "the",
        "and",
        "is",
        "in",
        "to",
        "of",
        "a",
        "for",
        "that",
        "on",
        "with",
        "as",
        "are",
        "by",
        "this",
        "an",
        "be",
        "or",
        "from",
        "at",
    }



class User(UserMixin):
    """Lightweight user wrapper for Flask-Login."""

    def __init__(self, document: Dict[str, Any]):
        self.id = document["_id"]
        self.username = document["username"]
        self.email = document["email"]
        self.password = document["password"]


class SimpleSummarizer:
    """TF-IDF + cosine-similarity TextRank summarizer that handles long PDFs well."""

    # Maximum sentences to process at once before chunked mode kicks in.
    _CHUNK_THRESHOLD = 300
    _CHUNK_SIZE = 150

    def __init__(self) -> None:
        try:
            self.stop_words = set(stopwords.words("english"))
        except Exception:
            self.stop_words = _nltk_stopwords

    def _tokens(self, text: str) -> List[str]:
        words = word_tokenize(text.lower()) if text.strip() else []
        return [w for w in words if w.isalpha() and w not in self.stop_words]

    def _sentences(self, text: str) -> List[str]:
        try:
            return [s.strip() for s in sent_tokenize(text) if s.strip()]
        except Exception:
            return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    def _score_sentences_tfidf(self, sentences: List[str]) -> Dict[int, float]:
        """Score each sentence using TF-IDF cosine similarity to the document centroid."""
        if len(sentences) < 2:
            return {0: 1.0}
        try:
            vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=8000)
            matrix = vec.fit_transform(sentences)
            # Centroid = mean of all sentence vectors
            centroid = matrix.mean(axis=0)
            from sklearn.metrics.pairwise import cosine_similarity as _cos
            scores = _cos(centroid, matrix)[0]
            return {i: float(scores[i]) for i in range(len(sentences))}
        except Exception:
            # Fallback: word frequency scoring
            all_tokens = []
            for s in sentences:
                all_tokens.extend(self._tokens(s))
            freq = Counter(all_tokens)
            max_f = max(freq.values()) if freq else 1
            result: Dict[int, float] = {}
            for i, s in enumerate(sentences):
                result[i] = sum(freq.get(w, 0) / max_f for w in self._tokens(s))
            return result

    def _summarize_chunk(self, sentences: List[str], ratio: float, offset: int) -> List[int]:
        """Return global indices of the top sentences from a chunk."""
        scores = self._score_sentences_tfidf(sentences)
        count = max(1, int(len(sentences) * ratio))
        top_local = sorted(scores, key=lambda i: scores[i], reverse=True)[:count]
        return [offset + i for i in sorted(top_local)]

    def summarize(self, text: str, length: str = "medium") -> Dict[str, Any]:
        """Generate summary, keywords, flashcards and highlighted indices."""
        sentences = self._sentences(text)
        if not sentences:
            return {"summary": "No readable text was found in this PDF.",
                    "highlighted_indices": [], "keywords": [], "flashcards": []}

        ratio = {"short": 0.08, "medium": 0.15, "long": 0.25}.get(length, 0.15)
        # Cap output sentences so summaries don't become walls of text
        max_sentences = {"short": 6, "medium": 14, "long": 26}.get(length, 14)

        # --- Chunked mode for long documents ---
        if len(sentences) > self._CHUNK_THRESHOLD:
            top_indices: List[int] = []
            for start in range(0, len(sentences), self._CHUNK_SIZE):
                chunk = sentences[start: start + self._CHUNK_SIZE]
                top_indices.extend(self._summarize_chunk(chunk, ratio, start))
            top_indices = sorted(set(top_indices))
        else:
            scores = self._score_sentences_tfidf(sentences)
            count = max(1, min(int(len(sentences) * ratio), max_sentences))
            top_indices = sorted(
                sorted(scores, key=lambda i: scores[i], reverse=True)[:count]
            )

        # Enforce max sentence cap
        if len(top_indices) > max_sentences:
            # Keep best-scoring ones up to cap
            all_scores = self._score_sentences_tfidf(sentences)
            top_indices = sorted(
                sorted(top_indices, key=lambda i: all_scores.get(i, 0), reverse=True)[:max_sentences]
            )

        summary_sentences = [sentences[i] for i in top_indices] if top_indices else [sentences[0]]

        # Group into readable paragraphs (3 sentences each)
        paragraphs: List[str] = []
        for i in range(0, len(summary_sentences), 3):
            paragraphs.append(" ".join(summary_sentences[i: i + 3]))
        summary = "\n\n".join(paragraphs)

        # Keywords via TF-IDF on full text
        all_tokens = self._tokens(text)
        freq = Counter(all_tokens)
        keywords = [
            w for w, _ in freq.most_common(30)
            if w.isalpha() and len(w) > 3 and w not in self.stop_words
        ][:12]

        flashcards = self._flashcards(sentences, freq, top_indices[:8])
        return {
            "summary": summary,
            "highlighted_indices": top_indices,
            "keywords": keywords,
            "flashcards": flashcards,
        }

    def _flashcards(self, sentences: List[str], frequencies: Counter, indices: List[int]) -> List[Dict[str, str]]:
        """Create cloze-style flashcards from the strongest summary sentences."""
        flashcards: List[Dict[str, str]] = []
        for index in indices[:5]:
            sentence = sentences[index]
            tokens = [
                t for t in re.findall(r"\b[a-zA-Z]{4,}\b", sentence.lower())
                if t not in self.stop_words
            ]
            if not tokens:
                fallback = re.search(r"\b[a-zA-Z]{4,}\b", sentence)
                key_word = fallback.group(0) if fallback else None
            else:
                key_word = max(tokens, key=lambda w: frequencies.get(w, 0))
            if not key_word:
                continue
            masked = re.sub(rf"\b{re.escape(key_word)}\b", "____", sentence, count=1, flags=re.IGNORECASE)
            front = masked if len(masked) <= 220 else masked[:217] + "..."
            back = sentence if len(sentence) <= 420 else sentence[:417] + "..."
            flashcards.append({"front": front, "back": back})
        return flashcards




class SimpleChatbot:
    """Multi-passage TF-IDF retrieval chatbot with calibrated confidence and smart fallback."""

    _TOP_K = 3
    # If best score stays below this, we admit we couldn't find a good answer.
    _CONFIDENCE_THRESHOLD = 0.18



    def __init__(self, text: str) -> None:
        self._raw_text = text
        self.doc_context = self._extract_doc_context(text)
        self.passages = self._build_passages(text)
        if self.passages:
            self.vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 3),
                max_features=20000,
                sublinear_tf=True,
            )
            self.matrix = self.vectorizer.fit_transform(self.passages)
        else:
            self.vectorizer = None
            self.matrix = None

    # ------------------------------------------------------------------ #
    # Document-context extraction                                          #
    # ------------------------------------------------------------------ #
    def _extract_doc_context(self, text: str) -> Dict[str, Any]:
        """Scan the first ~600 chars of text for common metadata signals."""
        header = text[:600]
        ctx: Dict[str, Any] = {}

        # Institution / organisation
        inst = re.search(
            r"(?:university|college|institute|school|research\s+centre)[^\n]{0,80}",
            header, re.I
        )
        ctx["institution"] = inst.group(0).strip() if inst else None

        # Academic year or date
        year = re.search(r"\b(20\d{2}[-–]\d{2,4}|20\d{2})\b", header)
        ctx["year"] = year.group(0) if year else None

        # Subject / course name (lines that look like a course code or subject)
        subj = re.search(
            r"(?:subject|course|unit|paper|module)\s*[:\-]?\s*([A-Za-z][^\n]{3,60})",
            header, re.I
        )
        ctx["subject"] = subj.group(1).strip() if subj else None

        # Detect if it's a question bank
        ctx["is_qbank"] = bool(re.search(
            r"q[\s\-]?bank|question\s+bank|question\s+paper",
            text[:300], re.I
        ))

        # Main subject area: count AI-related topic words in full text
        ai_topics = [
            "artificial intelligence", "machine learning", "deep learning",
            "neural network", "search algorithm", "natural language",
            "knowledge representation", "planning", "reasoning",
        ]
        ctx["ai_topics"] = [t for t in ai_topics if t in text.lower()]

        # Top 8 numbered items as the topic list
        items = re.findall(r"(?:^|\n)\s*\d{1,3}[.)]\s*(.+?)(?=\n|$)", text)
        ctx["topic_list"] = [re.sub(r"\s+", " ", i).strip() for i in items[:8] if i.strip()]

        return ctx

    # ------------------------------------------------------------------ #
    # Meta-question detection & answering                                  #
    # ------------------------------------------------------------------ #
    _META_PATTERNS: List[tuple] = [
        # (regex to match question, handler key)
        (r"\b(main\s+topic|subject|about|cover|covers|regarding)\b", "topic"),
        (r"\b(theme|overview|purpose|objective|goal)\b",              "topic"),
        (r"\b(author|written\s+by|created\s+by|who\s+made)\b",        "author"),
        (r"\b(institution|college|university|school|organisation|organization)\b", "institution"),
        (r"\b(year|when|date|published|academic)\b",                  "year"),
        (r"\b(question\s+bank|q[\s\-]?bank|type\s+of\s+document|kind\s+of)\b", "doctype"),
        (r"\b(topics?\s+covered|list\s+of\s+topics|topics?\s+include)\b", "topics"),
    ]

    def _try_meta_answer(self, question: str) -> Optional[Dict[str, Any]]:
        """Return a structured answer if this is a document meta-question."""
        q = question.lower()
        matched_keys: set = set()
        for pattern, key in self._META_PATTERNS:
            if re.search(pattern, q, re.I):
                matched_keys.add(key)

        if not matched_keys:
            return None

        ctx = self.doc_context
        lines: List[str] = []

        if "doctype" in matched_keys or "topic" in matched_keys:
            if ctx.get("is_qbank"):
                lines.append("📄 Document type: Question Bank / Exam Q-Paper")
            if ctx.get("ai_topics"):
                lines.append(f"📚 Subject area: Artificial Intelligence — covers {', '.join(ctx['ai_topics'][:3])}")
            elif ctx.get("subject"):
                lines.append(f"📚 Subject: {ctx['subject']}")
            else:
                lines.append("📚 Subject area: Artificial Intelligence (based on topic analysis)")

        if "author" in matched_keys:
            if ctx.get("institution"):
                lines.append(f"🏫 Published by: {ctx['institution']}")
            else:
                lines.append("🏫 No specific author is mentioned. This appears to be an institutional question bank.")

        if "institution" in matched_keys and ctx.get("institution"):
            lines.append(f"🏫 Institution: {ctx['institution']}")

        if "year" in matched_keys:
            if ctx.get("year"):
                lines.append(f"📅 Academic year: {ctx['year']}")
            else:
                lines.append("📅 No academic year detected in the document.")

        if ("topics" in matched_keys or "topic" in matched_keys) and ctx.get("topic_list"):
            lines.append("🗂 Key topics in this document:")
            for t in ctx["topic_list"][:6]:
                lines.append(f"  • {t}")

        if not lines:
            return None

        return {
            "answer": "\n".join(lines),
            "confidence": 0.95,
            "low_confidence": False,
        }

    def _build_passages(self, text: str) -> List[str]:
        """Build rich passage index: numbered items + sentence windows + paragraphs."""
        passages: List[str] = []
        seen_keys: set = set()

        def add(p: str) -> None:
            p = p.strip()
            key = p.lower()[:80]
            # Skip junk: too short, mostly redacted tokens, or pure numbers
            if len(p) < 30:
                return
            if re.fullmatch(r"[\d\s\.\-\[\]redacted_emailphoneurl]+", p, re.I):
                return
            if key not in seen_keys:
                seen_keys.add(key)
                passages.append(p)

        # 1. Extract numbered list items (e.g. "1. DFS in AI", "21. Adversarial Search...")
        #    Combine each item with its neighbours for context.
        numbered = re.findall(
            r"(?:^|\n)\s*(\d{1,3})[.)]\s*(.+?)(?=\n\s*\d{1,3}[.)]|\Z)",
            text, re.DOTALL
        )
        items = [(int(n), re.sub(r"\s+", " ", body).strip()) for n, body in numbered if body.strip()]
        for i, (_, body) in enumerate(items):
            # Single item
            add(body)
            # Item + next (for context)
            if i + 1 < len(items):
                add(body + " " + items[i + 1][1])

        # 2. Sliding sentence windows (stride-1, window=3)
        raw_sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 20]
        for i in range(len(raw_sents)):
            chunk = " ".join(raw_sents[i: i + 3])
            add(chunk)

        # 3. Paragraph blocks (good for definitions / explanations)
        for block in re.split(r"\n{2,}", text):
            add(block)

        return passages


    def _expand_query(self, question: str) -> str:
        """Add simple synonym expansion to boost recall on short questions."""
        expansions: Dict[str, List[str]] = {
            "who": ["person", "author", "name"],
            "what": ["definition", "meaning", "description"],
            "when": ["date", "year", "time", "period"],
            "where": ["location", "place", "country", "city"],
            "how": ["method", "process", "step", "procedure"],
            "why": ["reason", "cause", "purpose"],
        }
        words = question.lower().split()
        extra: List[str] = []
        for w in words:
            if w in expansions:
                extra.extend(expansions[w])
        return question + " " + " ".join(extra) if extra else question

    def _format_answer(self, passage: str, question_terms: set) -> str:
        """Clean and structure the retrieved passage for display."""
        formatted = passage.strip()
        # Strip common heading prefixes
        for heading in ("technical skills", "skills", "education", "experience",
                        "projects", "project", "summary", "objective", "contact"):
            if formatted.lower().startswith(heading):
                formatted = formatted[len(heading):].lstrip(" :-|\n")
                break
        # Format skill lists
        if {"skill", "skills"} & question_terms:
            for label in ("Programming Languages:", "Web Technologies:", "Databases:",
                          "Frameworks:", "Tools:", "Libraries:", "Platforms:"):
                formatted = formatted.replace(label, f"\n• {label}")
        formatted = re.sub(r"[ \t]+", " ", formatted)
        formatted = re.sub(r" *\n *", "\n", formatted)
        formatted = re.sub(r"\n{3,}", "\n\n", formatted).strip()
        return formatted

    def answer(self, question: str) -> Dict[str, Any]:
        """Retrieve top-k passages, merge, return answer + calibrated confidence."""
        # Try meta-question context first
        meta_res = self._try_meta_answer(question)
        if meta_res is not None:
            return meta_res

        if not self.passages or self.vectorizer is None:
            return {"answer": "No text is available for this PDF yet.", "confidence": 0.0}

        expanded = self._expand_query(question)
        query_matrix = self.vectorizer.transform([expanded])
        tfidf_scores = cosine_similarity(query_matrix, self.matrix)[0]

        query_terms = set(re.findall(r"\b[a-zA-Z]{3,}\b", question.lower())) - _nltk_stopwords
        if not query_terms:
            query_terms = set(re.findall(r"\b[a-zA-Z]{3,}\b", question.lower()))

        intent_boosts: Dict[str, set] = {
            "skills":     {"skills", "skill", "technical", "programming", "languages", "tools"},
            "education":  {"education", "degree", "college", "school", "university", "cgpa"},
            "projects":   {"project", "projects", "built", "implemented", "developed"},
            "experience": {"experience", "internship", "worked", "job", "role"},
            "contact":    {"contact", "email", "phone", "linkedin", "github"},
            "algorithm":  {"algorithm", "algo", "search", "bfs", "dfs", "minimax", "heuristic"},
            "ai":         {"artificial", "intelligence", "agent", "goal", "task", "domain"},
        }

        passage_scores: List[tuple] = []
        for idx, passage in enumerate(self.passages):
            p_lower = passage.lower()
            overlap = sum(1 for t in query_terms if t in p_lower)
            intent_bonus = sum(
                0.12 for trigger in intent_boosts.values()
                if query_terms & trigger and any(tok in p_lower for tok in trigger)
            )
            phrase_bonus = 0.25 if question.lower().rstrip("?").strip() in p_lower else 0.0
            # Penalise passages that look like bare question lists (no sentence verbs)
            is_question_list = bool(re.search(r"^\d+[.)]\s", passage, re.M)) and len(passage) < 200
            list_penalty = -0.10 if is_question_list else 0.0
            score = (float(tfidf_scores[idx])
                     + min(overlap * 0.14, 0.45)
                     + intent_bonus + phrase_bonus + list_penalty)
            passage_scores.append((score, idx))

        passage_scores.sort(reverse=True)
        best_raw_score = passage_scores[0][0] if passage_scores else 0.0

        # Low-confidence fallback — be honest rather than give irrelevant text
        if best_raw_score < self._CONFIDENCE_THRESHOLD:
            suggestions = [
                "Try rephrasing with more specific terms from the document.",
                "Ask about a specific topic, algorithm, or concept mentioned in the PDF.",
                "Example: 'Explain DFS' or 'What is hill climbing?'",
            ]
            return {
                "answer": (
                    "⚠️ I couldn't find a confident answer to that question in this document.\n\n"
                    + "\n".join(f"• {s}" for s in suggestions)
                ),
                "confidence": 0.0,
                "low_confidence": True,
            }

        top_k = passage_scores[:self._TOP_K]

        # Merge top-k — skip passages that are pure numbered question lists
        seen: set = set()
        merged_sentences: List[str] = []
        for _, idx in top_k:
            passage = self.passages[idx]
            # Skip if it looks like a bare numbered list with no explanation
            if re.match(r"^\d+[.)]\s+\w[\w\s]{0,40}$", passage.strip()):
                continue
            for sent in re.split(r"(?<=[.!?])\s+", passage):
                sent = sent.strip()
                key = sent.lower()[:60]
                if sent and key not in seen and len(sent) > 20:
                    seen.add(key)
                    merged_sentences.append(sent)

        if not merged_sentences:
            # Fallback: use the top passage directly
            merged_sentences = [self.passages[top_k[0][1]]]

        merged = " ".join(merged_sentences)
        answer = self._format_answer(merged, query_terms)

        if len(answer) > 750:
            answer = answer[:747].rsplit(" ", 1)[0] + "..."

        confidence = min(best_raw_score * 1.35, 1.0)
        return {"answer": answer, "confidence": round(confidence, 3)}




class Storage:
    """Tiny abstraction over MongoDB or in-memory demo storage."""

    @staticmethod
    def _demo_collection(name: str) -> Dict[str, Dict[str, Any]]:
        return DEMO_DB[name]
    @staticmethod
    def insert(collection: str, document: Dict[str, Any]) -> None:
        global DEMO_MODE
        if DEMO_MODE:
            Storage._demo_collection(collection)[document["_id"]] = document
            return
        try:
            mongo.db[collection].insert_one(document)
        except Exception:
            # Fall back to demo mode if MongoDB fails
            DEMO_MODE = True
            Storage._demo_collection(collection)[document["_id"]] = document
            create_default_demo_user()

    @staticmethod
    def update(collection: str, document_id: str, updates: Dict[str, Any]) -> None:
        global DEMO_MODE
        if DEMO_MODE:
            Storage._demo_collection(collection).setdefault(document_id, {}).update(updates)
            return
        try:
            mongo.db[collection].update_one({"_id": document_id}, {"$set": updates}, upsert=True)
        except Exception:
            # switch to demo mode on error
            DEMO_MODE = True
            Storage._demo_collection(collection).setdefault(document_id, {}).update(updates)
            create_default_demo_user()

    @staticmethod
    def find_one(collection: str, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        global DEMO_MODE
        if DEMO_MODE:
            for document in Storage._demo_collection(collection).values():
                if all(document.get(key) == value for key, value in query.items()):
                    return document
            return None
        try:
            return mongo.db[collection].find_one(query)
        except Exception:
            # switch to demo mode on error
            DEMO_MODE = True
            create_default_demo_user()
            for document in Storage._demo_collection(collection).values():
                if all(document.get(key) == value for key, value in query.items()):
                    return document
            return None

    @staticmethod
    def find(collection: str, query: Optional[Dict[str, Any]] = None, sort_key: Optional[str] = None, descending: bool = True) -> List[Dict[str, Any]]:
        query = query or {}
        global DEMO_MODE
        if DEMO_MODE:
            documents = [document for document in Storage._demo_collection(collection).values() if all(document.get(key) == value for key, value in query.items())]
            if sort_key:
                documents.sort(key=lambda item: item.get(sort_key) or datetime.min, reverse=descending)
            return documents
        try:
            cursor = mongo.db[collection].find(query)
            if sort_key:
                cursor = cursor.sort(sort_key, -1 if descending else 1)
            documents = list(cursor)
            # Ensure we never return None to callers
            return documents if documents is not None else []
        except Exception:
            DEMO_MODE = True
            create_default_demo_user()
            documents = [document for document in Storage._demo_collection(collection).values() if all(document.get(key) == value for key, value in query.items())]
            if sort_key:
                documents.sort(key=lambda item: item.get(sort_key) or datetime.min, reverse=descending)
            return documents if documents is not None else []

    @staticmethod
    def delete(collection: str, query: Dict[str, Any]) -> bool:
        """Delete one document matching query and return True if deleted."""

        global DEMO_MODE
        if DEMO_MODE:
            bucket = Storage._demo_collection(collection)
            for document_id, document in list(bucket.items()):
                if all(document.get(key) == value for key, value in query.items()):
                    del bucket[document_id]
                    return True
            return False
        try:
            result = mongo.db[collection].delete_one(query)
            return bool(result.deleted_count)
        except Exception:
            DEMO_MODE = True
            create_default_demo_user()
            bucket = Storage._demo_collection(collection)
            for document_id, document in list(bucket.items()):
                if all(document.get(key) == value for key, value in query.items()):
                    del bucket[document_id]
                    return True
            return False


def now_iso() -> str:
    """Return the current UTC timestamp as an ISO string."""

    return datetime.utcnow().isoformat()


def serialize_document(document: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Convert database objects into JSON-friendly values."""

    if not document:
        return None
    result = dict(document)
    for key, value in list(result.items()):
        if isinstance(value, datetime):
            result[key] = value.isoformat()
        elif isinstance(value, ObjectId):
            result[key] = str(value)
    return result


def create_default_demo_user() -> Dict[str, Any]:
    """Create a default demo user so the UI can be explored without MongoDB."""

    user = Storage.find_one("users", {"email": "demo@local"})
    if user:
        return user
    user = {"_id": "demo-user", "username": "Demo User", "email": "demo@local", "password": generate_password_hash("demo1234")}
    Storage.insert("users", user)
    return user


@login_manager.user_loader
def load_user(user_id: str) -> Optional[User]:
    """Load a user from MongoDB or demo storage."""

    return User(Storage.find_one("users", {"_id": user_id})) if Storage.find_one("users", {"_id": user_id}) else None


def login_demo_user_if_needed() -> None:
    """Allow UI browsing in demo mode without requiring explicit login."""

    if DEMO_MODE and not current_user.is_authenticated and request.endpoint not in {"login", "signup", "static"}:
        login_user(User(create_default_demo_user()))


@app.before_request
def before_request() -> None:
    """Set up demo access before every request."""

    login_demo_user_if_needed()


@app.context_processor
def inject_globals() -> Dict[str, Any]:
    """Expose shared values to templates."""

    return {"demo_mode": DEMO_MODE, "current_year": datetime.utcnow().year}


def test_mongo_connection() -> bool:
    """Return True if MongoDB responds to a ping, otherwise enable demo mode."""

    global DEMO_MODE
    try:
        mongo.cx.admin.command("ping")
        DEMO_MODE = False
        return True
    except Exception:
        DEMO_MODE = True
        create_default_demo_user()
        return False


# Run this at import time so both `python app.py` and `flask run`
# reflect the real MongoDB state.
test_mongo_connection()


def extract_pdf_text(file_path: str) -> Dict[str, Any]:
    """Extract text and page count from a PDF file."""

    page_texts: List[str] = []
    page_count = 0
    try:
        with pdfplumber.open(file_path) as pdf:
            page_count = len(pdf.pages)
            for page in pdf.pages:
                page_texts.append(page.extract_text() or "")
    except Exception:
        pass
    raw = "\n".join(page_texts).strip()
    # Basic sanitization: keep paragraph structure while removing bad chars.
    cleaned = raw.replace("|", "\n")
    cleaned = re.sub(r"[\x00-\x08\x0B-\x1F\x7F]+", " ", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    # Aggressive redaction for privacy: redact emails, phone numbers, and URLs
    cleaned = re.sub(r"\b[\w.%-]+@[\w.-]+\.[A-Za-z]{2,}\b", "[redacted_email]", cleaned)
    cleaned = re.sub(r"https?://\S+|www\.\S+", "[redacted_url]", cleaned)
    # Simple phone number patterns (international and local) - replace with token
    cleaned = re.sub(r"\+?\d[\d\s\-()]{6,}\d", "[redacted_phone]", cleaned)

    # Remove standalone short tokens that often appear from CVs/contacts
    cleaned = re.sub(r"\b(?:com|www|linkedin|github|mailto)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    return {"text": cleaned, "page_count": page_count}


def process_pdf_background(pdf_id: str, file_path: str) -> None:
    """Process an uploaded PDF in a background thread."""

    with app.app_context():
        try:
            Storage.update("pdfs", pdf_id, {"status": "processing"})
            extracted = extract_pdf_text(file_path)
            Storage.update(
                "pdfs",
                pdf_id,
                {
                    "text": extracted["text"],
                    "page_count": extracted["page_count"],
                    "status": "done",
                    "updated_at": now_iso(),
                },
            )
        except Exception as exc:
            Storage.update("pdfs", pdf_id, {"status": "error", "error": str(exc), "updated_at": now_iso()})


def get_pdf_or_404(pdf_id: str) -> Optional[Dict[str, Any]]:
    """Fetch a PDF document for the current user."""

    return Storage.find_one("pdfs", {"_id": pdf_id, "user_id": current_user.id})


def get_summary_document(pdf_id: str) -> Optional[Dict[str, Any]]:
    """Fetch an existing summary for a PDF."""

    return Storage.find_one("summaries", {"_id": pdf_id, "user_id": current_user.id})


@app.route("/")
def index() -> Response:
    """Send users to the dashboard or login page."""

    return redirect(url_for("dashboard" if current_user.is_authenticated or DEMO_MODE else "login"))


@app.route("/signup", methods=["GET", "POST"])
def signup() -> Response:
    """Render signup page or create a new account."""

    if request.method == "GET":
        return render_template("signup.html")
    payload = request.get_json(silent=True) or request.form
    username = (payload.get("username") or "").strip()
    email = (payload.get("email") or "").strip().lower()
    password = payload.get("password") or ""
    if not username or not email or len(password) < 6:
        return jsonify({"success": False, "message": "Please provide a valid name, email, and password."}), 400
    if Storage.find_one("users", {"email": email}):
        return jsonify({"success": False, "message": "An account with that email already exists."}), 409
    user = {"_id": str(uuid.uuid4()), "username": username, "email": email, "password": generate_password_hash(password)}
    Storage.insert("users", user)
    login_user(User(user))
    return jsonify({"success": True, "message": "Account created successfully.", "redirect": url_for("dashboard")})


@app.route("/login", methods=["GET", "POST"])
def login() -> Response:
    """Render login page or authenticate an existing user."""

    if request.method == "GET":
        return render_template("login.html")
    payload = request.get_json(silent=True) or request.form
    email = (payload.get("email") or "").strip().lower()
    password = payload.get("password") or ""
    user = Storage.find_one("users", {"email": email})
    if not user or not check_password_hash(user["password"], password):
        return jsonify({"success": False, "message": "Invalid email or password."}), 401
    login_user(User(user))
    return jsonify({"success": True, "message": "Login successful.", "redirect": url_for("dashboard")})


@app.route("/logout")
@login_required
def logout() -> Response:
    """Log the user out and return to the login page."""

    logout_user()
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard() -> Response:
    """Show recent uploads, summaries, and project statistics."""

    pdfs = Storage.find("pdfs", {"user_id": current_user.id}, sort_key="upload_date") or []
    summaries = Storage.find("summaries", {"user_id": current_user.id}, sort_key="updated_at") or []
    month_key = datetime.utcnow().strftime("%Y-%m")
    this_month = [pdf for pdf in pdfs if str(pdf.get("upload_date", "")).startswith(month_key)]
    return render_template(
        "dashboard.html",
        pdf_count=len(pdfs),
        summary_count=len(summaries),
        month_count=len(this_month),
        recent_pdfs=pdfs[:5],
        recent_summaries=summaries[:5],
    )


@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload() -> Response:
    """Render the upload page or accept one or more PDF files."""

    if request.method == "GET":
        return render_template("upload.html")
    files = request.files.getlist("files") or request.files.getlist("pdfs")
    if not files:
        return jsonify({"success": False, "message": "Please choose at least one PDF file."}), 400
    
    uploaded: List[Dict[str, Any]] = []
    try:
        for file_storage in files:
            original_name = file_storage.filename or ""
            if not original_name.lower().endswith(".pdf"):
                continue
                
            safe_name = secure_filename(original_name)
            if not safe_name or not safe_name.lower().endswith(".pdf"):
                safe_name = f"document_{uuid.uuid4().hex[:6]}.pdf"
                
            pdf_id = str(uuid.uuid4())
            stored_name = f"{pdf_id}.pdf"
            file_path = BASE_DIR / stored_name
            file_storage.save(file_path)
            
            # Extract text
            extracted = extract_pdf_text(str(file_path))
            status = "done" if extracted.get("text") else "queued"
            
            record = {
                "_id": pdf_id,
                "user_id": current_user.id,
                "filename": stored_name,
                "original_name": original_name,
                "filepath": str(file_path),
                "file_size": file_path.stat().st_size if file_path.exists() else 0,
                "page_count": int(extracted.get("page_count", 0)),
                "status": status,
                "upload_date": now_iso(),
                "text": extracted.get("text", ""),
            }
            Storage.insert("pdfs", record)
            
            # If extraction yielded no text, process it. 
            # On Vercel, we must do this synchronously because background threads are frozen.
            if not extracted.get("text"):
                if os.environ.get("VERCEL"):
                    process_pdf_background(pdf_id, str(file_path))
                else:
                    threading.Thread(target=process_pdf_background, args=(pdf_id, str(file_path)), daemon=True).start()
                    
            uploaded.append(serialize_document(record) or record)
            
        return jsonify({"success": True, "message": f"Uploaded {len(uploaded)} PDF file(s).", "files": uploaded})
    except Exception as e:
        app.logger.error(f"Upload error: {str(e)}")
        return jsonify({"success": False, "message": f"Server processing error: {str(e)}"}), 500


@app.route("/summarize/<pdf_id>")
@login_required
def summarize_page(pdf_id: str) -> Response:
    """Render the summary workspace for a selected PDF."""

    pdf = get_pdf_or_404(pdf_id)
    if not pdf:
        flash("PDF not found.", "danger")
        return redirect(url_for("dashboard"))
    summary = get_summary_document(pdf_id)
    return render_template("summary.html", pdf=pdf, summary=summary)


@app.route("/chat/<pdf_id>")
@login_required
def chat_page(pdf_id: str) -> Response:
    """Render the question-answering page for a selected PDF."""

    pdf = get_pdf_or_404(pdf_id)
    if not pdf:
        flash("PDF not found.", "danger")
        return redirect(url_for("dashboard"))
    return render_template("chat.html", pdf=pdf)


@app.route("/api/pdfs")
@login_required
def api_pdfs() -> Response:
    """Return the current user's uploaded PDFs as JSON."""
    pdfs = Storage.find("pdfs", {"user_id": current_user.id}, sort_key="upload_date") or []
    return jsonify({"success": True, "pdfs": [serialize_document(pdf) for pdf in pdfs]})


@app.route("/api/summaries")
@login_required
def api_summaries() -> Response:
    """Return the current user's summaries as JSON."""
    summaries = Storage.find("summaries", {"user_id": current_user.id}, sort_key="updated_at") or []
    return jsonify({"success": True, "summaries": [serialize_document(summary) for summary in summaries]})


@app.route("/api/summary/<pdf_id>", methods=["GET", "POST", "DELETE"])
@login_required
def api_summary(pdf_id: str) -> Response:
    """Fetch or generate a summary for one PDF."""

    pdf = get_pdf_or_404(pdf_id)
    if not pdf:
        return jsonify({"success": False, "message": "PDF not found."}), 404
    if request.method == "GET":
        summary = get_summary_document(pdf_id)
        return jsonify({"success": True, "summary": serialize_document(summary)})
    if request.method == "DELETE":
        deleted = Storage.delete("summaries", {"_id": pdf_id, "user_id": current_user.id})
        if not deleted:
            return jsonify({"success": False, "message": "Summary not found."}), 404
        return jsonify({"success": True, "message": "Summary deleted successfully."})
    payload = request.get_json(silent=True) or request.form
    length = (payload.get("length") or "medium").lower()
    if not pdf.get("text"):
        # Try a quick synchronous extraction in case background processing
        # hasn't completed yet (helps small PDFs and reduces race conditions).
        try:
            extracted = extract_pdf_text(pdf.get("filepath", ""))
            if extracted.get("text"):
                Storage.update("pdfs", pdf_id, {"text": extracted.get("text", ""), "page_count": int(extracted.get("page_count", 0)), "status": "done", "updated_at": now_iso()})
                pdf = get_pdf_or_404(pdf_id)
        except Exception:
            pass
    if not pdf.get("text"):
        return jsonify({"success": False, "message": "Text extraction is still running. Try again in a moment."}), 409
    summarizer = SimpleSummarizer()
    result = summarizer.summarize(pdf["text"], length)
    summary = {
        "_id": pdf_id,
        "pdf_id": pdf_id,
        "user_id": current_user.id,
        "pdf_name": pdf.get("original_name", "PDF"),
        "length": length,
        "summary_text": result["summary"],
        "highlighted_indices": result["highlighted_indices"],
        "keywords": result["keywords"],
        "flashcards": result["flashcards"],
        "updated_at": now_iso(),
    }
    existing = get_summary_document(pdf_id)
    if existing:
        summary["created_at"] = existing.get("created_at", now_iso())
    else:
        summary["created_at"] = now_iso()
    Storage.update("summaries", pdf_id, summary)
    return jsonify({"success": True, "summary": serialize_document(summary)})


@app.route("/api/pdf/<pdf_id>", methods=["DELETE"])
@login_required
def api_delete_pdf(pdf_id: str) -> Response:
    """Delete a PDF record, its summary, and the file from disk."""

    pdf = get_pdf_or_404(pdf_id)
    if not pdf:
        return jsonify({"success": False, "message": "PDF not found."}), 404

    # Remove the physical file from disk, ignoring missing-file errors.
    file_path = Path(pdf.get("filepath", ""))
    try:
        if file_path.exists():
            file_path.unlink()
    except Exception as exc:
        # Log but don't block deletion of DB records.
        app.logger.warning("Could not delete file %s: %s", file_path, exc)

    # Delete summary (if any) first, then the PDF record.
    Storage.delete("summaries", {"_id": pdf_id, "user_id": current_user.id})
    Storage.delete("pdfs", {"_id": pdf_id, "user_id": current_user.id})

    return jsonify({"success": True, "message": "PDF and its summary deleted successfully."})


@app.route("/api/chat", methods=["POST"])
@login_required
def api_chat() -> Response:
    """Answer a question using the selected PDF text."""

    payload = request.get_json(silent=True) or {}
    pdf_id = payload.get("pdf_id")
    question = (payload.get("question") or "").strip()
    if not pdf_id or not question:
        return jsonify({"success": False, "message": "Both pdf_id and question are required."}), 400
    pdf = get_pdf_or_404(pdf_id)
    if not pdf or not pdf.get("text"):
        return jsonify({"success": False, "message": "PDF text is not available yet."}), 409
    bot = SimpleChatbot(pdf["text"])
    return jsonify({"success": True, "response": bot.answer(question)})


@app.route("/api/export/<pdf_id>/txt")
@login_required
def export_txt(pdf_id: str) -> Response:
    """Download the generated summary as a plain-text file."""

    summary = get_summary_document(pdf_id)
    if not summary:
        return jsonify({"success": False, "message": "Summary not found."}), 404
    content = summary.get("summary_text", "")
    return Response(content, mimetype="text/plain", headers={"Content-Disposition": f'attachment; filename="{pdf_id}.txt"'})


@app.route("/api/export/<pdf_id>/json")
@login_required
def export_json(pdf_id: str) -> Response:
    """Download the summary as JSON."""

    summary = get_summary_document(pdf_id)
    if not summary:
        return jsonify({"success": False, "message": "Summary not found."}), 404
    return Response(json.dumps(serialize_document(summary), indent=2), mimetype="application/json", headers={"Content-Disposition": f'attachment; filename="{pdf_id}.json"'})


if __name__ == "__main__":
    test_mongo_connection()
    # Disable the auto-reloader on Windows to avoid intermittent
    # OSError (WinError 10038) during rapid restarts in threaded mode.
    app.run(debug=True, use_reloader=False)
