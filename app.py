from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from groq import Groq
import os
import base64
from supabase import create_client
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_IMAGE_TYPES = {'image/jpeg', 'image/png', 'image/gif', 'image/webp'}

app = Flask(__name__)
CORS(app)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32 MB upload limit

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
db = create_client(supabase_url, supabase_key)


# ── helpers ──────────────────────────────────────────────────────────────────

def ask_ai(prompt, system="You are a helpful study assistant."):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


def extract_text_from_docx(filepath):
    """Extract plain text from a .docx file."""
    try:
        import docx
        doc = docx.Document(filepath)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except ImportError:
        raise RuntimeError("python-docx not installed. Run: pip install python-docx")


def extract_text_from_txt(filepath):
    """Read a plain-text or markdown file."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


# ── routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


# ---------- PDF upload + chunk-based QA ----------

@app.route("/upload", methods=["POST"])
def upload_file():
    """Upload a PDF, split into chunks, return chunks + auto summary."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        loader = PyPDFLoader(filepath)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(pages)
        chunks = [doc.page_content for doc in docs]
        return jsonify({"message": f"Successfully indexed {filename}!", "chunks": chunks})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ask-pdf", methods=["POST"])
def ask_pdf():
    question = request.json.get("question")
    chunks   = request.json.get("chunks", [])
    if not chunks:
        return jsonify({"answer": "No PDF uploaded yet. Please upload a PDF first."})
    context = "\n\n".join(chunks[:6])
    prompt  = f"Based on this document:\n\n{context}\n\nAnswer this question: {question}"
    answer  = ask_ai(
        prompt,
        "You are a helpful study assistant. Answer based on the provided document content only. "
        "Use markdown formatting with headers, bullet points, and bold text where appropriate."
    )
    return jsonify({"answer": answer})


# ---------- DOCX / TXT upload + QA ----------

@app.route("/upload-doc", methods=["POST"])
def upload_doc():
    """Upload a .docx or .txt file and return its text content."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''

    try:
        if ext == 'docx':
            content = extract_text_from_docx(filepath)
        else:
            content = extract_text_from_txt(filepath)
        return jsonify({"content": content, "filename": filename})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ask-doc", methods=["POST"])
def ask_doc():
    """Answer a question based on previously uploaded DOCX/TXT content."""
    data     = request.json
    question = data.get("question", "Summarize this document.")
    content  = data.get("content", "")
    filename = data.get("filename", "document")
    if not content:
        return jsonify({"answer": "No document content provided."}), 400

    truncated = content[:12000]
    prompt    = f"Document: {filename}\n\n{truncated}\n\nQuestion: {question}"
    answer    = ask_ai(
        prompt,
        "You are a helpful assistant. Answer questions based on the provided document content. "
        "Use markdown formatting where appropriate."
    )
    return jsonify({"answer": answer})


# ---------- Image analysis ----------

@app.route("/ask-image", methods=["POST"])
def ask_image():
    """Analyse a base64-encoded image and answer the user's question about it."""
    data       = request.json
    question   = data.get("question", "Describe this image in detail.")
    image_data = data.get("image_data")
    mime_type  = data.get("mime_type", "image/jpeg")

    if not image_data:
        return jsonify({"answer": "No image data provided."}), 400

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{image_data}"}
                        },
                        {"type": "text", "text": question}
                    ]
                }
            ],
            max_tokens=1024
        )
        return jsonify({"answer": response.choices[0].message.content})
    except Exception as e:
        return jsonify({"answer": f"Error processing image: {str(e)}"}), 500


# ---------- Universal file upload (PDF / DOCX / TXT / image) ----------

@app.route("/upload-any", methods=["POST"])
def upload_any():
    """
    Single endpoint that accepts any supported file type and returns
    either chunked text (PDF), plain text (DOCX/TXT), or a base64
    image payload ready for /ask-image.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename  = secure_filename(file.filename)
    filepath  = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    ext       = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    mime_type = file.content_type or ''

    try:
        # ── Image ──────────────────────────────────────────────────────────
        if mime_type in ALLOWED_IMAGE_TYPES or ext in ('jpg', 'jpeg', 'png', 'gif', 'webp'):
            with open(filepath, 'rb') as f:
                b64 = base64.b64encode(f.read()).decode('utf-8')
            detected_mime = mime_type if mime_type in ALLOWED_IMAGE_TYPES else f"image/{ext}"
            return jsonify({"type": "image", "image_data": b64, "mime_type": detected_mime, "filename": filename})

        # ── PDF ────────────────────────────────────────────────────────────
        if ext == 'pdf':
            loader  = PyPDFLoader(filepath)
            pages   = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs    = splitter.split_documents(pages)
            chunks  = [doc.page_content for doc in docs]
            return jsonify({"type": "pdf", "chunks": chunks, "filename": filename})

        # ── DOCX ───────────────────────────────────────────────────────────
        if ext == 'docx':
            content = extract_text_from_docx(filepath)
            return jsonify({"type": "doc", "content": content, "filename": filename})

        # ── Plain text / markdown ──────────────────────────────────────────
        if ext in ('txt', 'md', 'csv'):
            content = extract_text_from_txt(filepath)
            return jsonify({"type": "doc", "content": content, "filename": filename})

        return jsonify({"error": f"Unsupported file type: .{ext}"}), 415

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------- General chat ----------

@app.route("/chat", methods=["POST"])
def chat():
    question = request.json.get("question")
    try:
        answer = ask_ai(
            question,
            "You are a helpful AI assistant. Answer any question clearly. "
            "Use markdown formatting with headers, bullet points, and bold text where appropriate."
        )
        db.table("messages").insert({"role": "user",    "content": question}).execute()
        db.table("messages").insert({"role": "ai",      "content": answer}).execute()
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"answer": f"Error: {str(e)}"})


# ---------- Study tools ----------

@app.route("/summarize", methods=["POST"])
def summarize():
    notes  = request.json.get("notes")
    answer = ask_ai(
        notes,
        "Summarize these notes into: 1) A short overview, 2) Key concepts as bullet points, "
        "3) Important terms to remember."
    )
    return jsonify({"answer": answer})


@app.route("/flashcards", methods=["POST"])
def flashcards():
    topic  = request.json.get("topic")
    count  = request.json.get("count", 5)
    prompt = (
        f"Create {count} flashcards about: {topic}. "
        'Reply ONLY with a JSON array like: [{"q":"question","a":"answer"}]. No extra text.'
    )
    answer = ask_ai(prompt)
    return jsonify({"answer": answer})


@app.route("/studyplan", methods=["POST"])
def studyplan():
    subject = request.json.get("subject")
    days    = request.json.get("days", 7)
    goal    = request.json.get("goal", "general mastery")
    prompt  = f"Subject: {subject}\nDays: {days}\nGoal: {goal}\nCreate a day-by-day study plan."
    answer  = ask_ai(prompt, "You are a study planner. Create a clear, realistic day-by-day study plan.")
    return jsonify({"answer": answer})


# ---------- History ----------

@app.route("/history", methods=["GET"])
def history():
    data = db.table("messages").select("*").order("id").limit(20).execute()
    return jsonify({"messages": data.data})


@app.route("/clear", methods=["POST"])
def clear():
    db.table("messages").delete().neq("id", 0).execute()
    return jsonify({"status": "cleared"})


if __name__ == "__main__":
    app.run(debug=True)