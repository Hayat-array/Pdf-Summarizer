# Smart PDF Summarizer - College Project

A full-stack web application that uses Flask, MongoDB, and AI-powered NLP to summarize PDFs, generate flashcards, and answer questions through an intelligent chatbot.

## Project Overview

**Smart PDF Summarizer** is an end-to-end application that lets students:
- Upload PDF documents
- Generate automatic summaries (short/medium/long)
- Extract keywords from documents
- Create flashcards for study
- Ask questions about PDF content via a chatbot
- Export summaries as TXT or JSON

## Tech Stack

- **Backend**: Flask + Flask-PyMongo + Flask-Login
- **Database**: MongoDB
- **PDF Processing**: pdfplumber, PyPDF2
- **NLP**: NLTK, scikit-learn (TF-IDF)
- **Frontend**: HTML5 + Vanilla JavaScript + CSS3
- **Authentication**: Werkzeug password hashing

## Features Implemented

✅ User authentication (signup/login/logout)
✅ MongoDB integration with demo fallback mode
✅ PDF upload with background processing (threading)
✅ Text extraction from PDFs
✅ Automatic summarization with configurable length
✅ Keyword extraction
✅ Flashcard generation (cloze deletion style)
✅ TF-IDF based Q&A chatbot
✅ Export summaries (TXT + JSON)
✅ Responsive design (mobile-friendly)
✅ Dark/light theme toggle
✅ Session management

## Project Structure

```
mart_pdf/
├── app.py                 # Main Flask application (600+ lines)
├── config.py             # Configuration and environment setup
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variables template
├── static/
│   ├── css/
│   │   └── style.css     # Modern, responsive styling
│   └── js/
│       └── app.js        # Client-side logic (theme, forms, chat, uploads)
└── templates/
    ├── base.html         # Shared layout with sidebar
    ├── login.html        # Authentication
    ├── signup.html       # User registration
    ├── dashboard.html    # Overview and recent activity
    ├── upload.html       # Drag & drop PDF upload
    ├── summary.html      # Summarization & flashcards
    └── chat.html         # Q&A interface
```

## Installation & Setup

### 1. Prerequisites
- Python 3.8+
- MongoDB (local or Atlas cloud)
- pip or conda

### 2. Clone & Install Dependencies

```bash
cd mart_pdf
pip install -r requirements.txt
```

### 3. Configure Environment

Create a `.env` file from `.env.example`:

```bash
cp .env.example .env
```

Edit `.env` and add your MongoDB URI:

```
MONGO_URI=mongodb://localhost:27017/smart_pdf_summarizer
FLASK_SECRET_KEY=your-secret-key-here
```

### 4. Run the Application

```bash
python app.py
```

Or with Flask CLI:

```bash
flask run
```

The app will start on `http://localhost:5000`

**Demo Mode**: If MongoDB is unavailable, the app automatically enables demo mode with an in-memory database and a default demo user (email: `demo@local`, password: `demo1234`).

## Key Code Sections

### Authentication System
- Hashed password storage with `generate_password_hash()` / `check_password_hash()`
- Flask-Login session management
- Optional demo mode for testing without MongoDB

### PDF Processing
```python
# Background text extraction using threading
threading.Thread(target=process_pdf_background, args=(pdf_id, file_path), daemon=True).start()
```

### Summarization Engine
```python
class SimpleSummarizer:
    - Sentence tokenization
    - Word frequency scoring (TF-like)
    - Configurable summary lengths (10%, 20%, 30%)
    - Keyword extraction
    - Flashcard generation with cloze deletion
```

### Chatbot Q&A
```python
class SimpleChatbot:
    - TF-IDF vectorization of PDF sentences
    - Cosine similarity matching for query answering
    - Confidence score calculation
```

## API Endpoints

### Authentication
- `POST /signup` - User registration
- `POST /login` - User login
- `GET /logout` - Session logout

### Pages
- `GET /` - Redirect to dashboard/login
- `GET /dashboard` - Main dashboard
- `GET /upload` - Upload page
- `GET /summarize/<pdf_id>` - Summary workspace
- `GET /chat/<pdf_id>` - Chat interface

### API Routes (JSON)
- `GET /api/pdfs` - List user's PDFs
- `GET /api/summaries` - List user's summaries
- `GET /api/summary/<pdf_id>` - Get/generate summary
- `POST /api/chat` - Ask question (request: `{pdf_id, question}`)
- `GET /api/export/<pdf_id>/txt` - Download summary as TXT
- `GET /api/export/<pdf_id>/json` - Download summary as JSON

## Frontend Features

### JavaScript (app.js)
- **Theme Toggle**: Save dark/light preference to localStorage
- **Toast Notifications**: User feedback system
- **AJAX Forms**: Seamless auth/upload without page reload
- **Drag & Drop**: File upload with visual feedback
- **Chat Interface**: Real-time Q&A with confidence scores
- **Flashcard Flip**: Click-to-reveal study cards

### CSS Styling
- Modern gradient design with CSS variables
- Fixed sidebar navigation
- Responsive grid layouts
- Smooth animations & transitions
- Dark/light theme support
- Mobile-optimized (320px+)

## Database Schema

### users collection
```json
{
  "_id": "uuid",
  "username": "string",
  "email": "string (unique)",
  "password": "hashed string"
}
```

### pdfs collection
```json
{
  "_id": "uuid",
  "user_id": "uuid",
  "filename": "string",
  "original_name": "string",
  "filepath": "string",
  "file_size": "number",
  "page_count": "number",
  "status": "queued|processing|done|error",
  "text": "string (extracted text)",
  "upload_date": "ISO string",
  "updated_at": "ISO string"
}
```

### summaries collection
```json
{
  "_id": "pdf_id",
  "pdf_id": "string",
  "user_id": "string",
  "pdf_name": "string",
  "length": "short|medium|long",
  "summary_text": "string",
  "keywords": ["array of strings"],
  "flashcards": [
    {"front": "cloze question", "back": "answer"}
  ],
  "created_at": "ISO string",
  "updated_at": "ISO string"
}
```

## For College Projects

This codebase is designed for learning:

✅ **Clean, documented code** with docstrings on all classes/functions
✅ **Modular design** separating concerns (auth, storage, NLP, API routes)
✅ **Real-world patterns** (background jobs, error handling, storage abstraction)
✅ **No heavy frameworks** (no SQLAlchemy, no Celery, no React/Vue)
✅ **Lightweight ML** (NLTK + scikit-learn, no transformers)
✅ **Production considerations** (demo mode, password hashing, session management)

### Learning Outcomes

- Flask web framework & routing
- MongoDB document database
- User authentication & sessions
- File uploads & background processing
- NLP basics (tokenization, TF-IDF, similarity)
- Frontend DOM manipulation & AJAX
- Responsive CSS design
- Error handling & validation

## Troubleshooting

### MongoDB Connection Failed
The app will automatically enable demo mode. All features work but data is in-memory (lost on restart).

### ModuleNotFoundError: No module named 'pdfplumber'
```bash
pip install pdfplumber
```

### Port 5000 Already in Use
```bash
flask run --port 5001
```

### NLTK Data Corruption
If you encounter NLTK zip file errors, the app gracefully falls back to regex-based sentence splitting.

## Future Enhancements

- Multi-language support
- Advanced ML models (spaCy, transformers)
- Celery task queue for large files
- WebSocket real-time chat
- User subscription plans
- Document sharing & collaboration
- Full-text search across PDFs
- Export to PDF/DOCX with formatting

## License

This is a college project. Feel free to modify and learn from it!

---

**Built with ❤️ for learning**
