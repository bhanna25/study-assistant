from flask import Flask, request, jsonify, render_template, Response, stream_with_context, session
from flask_cors import CORS
from groq import Groq
import os
import base64
import json
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
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-in-prod")

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
db = create_client(supabase_url, supabase_key)


# ── helpers ──────────────────────────────────────────────────────────────────

def stream_ai(prompt, system="You are a helpful study assistant."):
    """Generator that yields SSE chunks from Groq streaming API."""
    stream = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt}
        ],
        stream=True
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield f"data: {json.dumps({'token': delta})}\n\n"
    yield "data: [DONE]\n\n"


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


# ── Auth routes ───────────────────────────────────────────────────────────────

@app.route("/auth/signup", methods=["POST"])
def auth_signup():
    data     = request.json
    name     = data.get("name", "").strip()
    email    = data.get("email", "").strip()
    password = data.get("password", "")

    if not name or not email or not password:
        return jsonify({"error": "All fields are required."}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters."}), 400

    try:
        res = db.auth.sign_up({"email": email, "password": password, "options": {"data": {"name": name}}})
        if res.user:
            return jsonify({"message": "Account created! Please login."})
        return jsonify({"error": "Signup failed. Try again."}), 400
    except Exception as e:
        err = str(e)
        if "already registered" in err.lower() or "already been registered" in err.lower():
            return jsonify({"error": "Email already registered."}), 400
        return jsonify({"error": err}), 400


@app.route("/auth/login", methods=["POST"])
def auth_login():
    data     = request.json
    email    = data.get("email", "").strip()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({"error": "Email and password required."}), 400

    try:
        res = db.auth.sign_in_with_password({"email": email, "password": password})
        if res.user:
            user_name = (res.user.user_metadata or {}).get("name") or email
            session["user"] = {"id": res.user.id, "email": res.user.email, "name": user_name}
            return jsonify({"user": session["user"]})
        return jsonify({"error": "Invalid credentials."}), 401
    except Exception as e:
        return jsonify({"error": "Invalid email or password."}), 401


@app.route("/auth/logout", methods=["POST"])
def auth_logout():
    session.pop("user", None)
    try:
        db.auth.sign_out()
    except Exception:
        pass
    return jsonify({"message": "Logged out."})


@app.route("/auth/me", methods=["GET"])
def auth_me():
    user = session.get("user")
    if user:
        return jsonify({"user": user})
    return jsonify({"user": None}), 401


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
    context = "\n\n".join(chunks[:8])
    prompt  = f"""DOCUMENT CONTENT:
    {context}

    USER REQUEST : {question}

    STRICT RULES:
- Answer ONLY using content from the document above
- Do NOT add any introduction, overview, or explanation unless asked
- If user asks for fill-in-the-blanks, generate ONLY fill-in-the-blank questions from the document
- If user asks for MCQs, generate ONLY MCQs from the document
- Match EXACTLY what the user asked for, nothing more"""
    answer  = ask_ai(
        prompt,
        "You are a strict document assistant. Do exactly what the user asks using only the document content. No overviews unless asked.")
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


# ---------- Streaming endpoints ----------

@app.route("/stream-chat", methods=["POST"])
def stream_chat():
    question = request.json.get("question", "")
    system = ("You are a helpful AI assistant. Answer any question clearly. "
              "Use markdown formatting with headers, bullet points, and bold text where appropriate.")
    try:
        db.table("messages").insert({"role": "user", "content": question}).execute()
    except Exception:
        pass
    return Response(stream_with_context(stream_ai(question, system)),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/stream-pdf", methods=["POST"])
def stream_pdf():
    data     = request.json
    question = data.get("question", "")
    chunks   = data.get("chunks", [])
    if not chunks:
        def no_chunks():
            yield 'data: {"token": "No PDF uploaded yet. Please attach a PDF first."}\n\n'
            yield "data: [DONE]\n\n"
        return Response(stream_with_context(no_chunks()), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    context = "\n\n".join(chunks[:6])
    prompt  = f"Based on this document:\n\n{context}\n\nAnswer this question: {question}"
    system  = ("You are a helpful study assistant. Answer based on the provided document content only. "
               "Use markdown formatting with headers, bullet points, and bold text where appropriate.")
    return Response(stream_with_context(stream_ai(prompt, system)),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/stream-doc", methods=["POST"])
def stream_doc():
    data     = request.json
    question = data.get("question", "")
    content  = data.get("content", "")
    filename = data.get("filename", "document")
    if not content:
        def no_content():
            yield 'data: {"token": "No document content provided."}\n\n'
            yield "data: [DONE]\n\n"
        return Response(stream_with_context(no_content()), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
    truncated = content[:12000]
    prompt    = f"Document: {filename}\n\n{truncated}\n\nQuestion: {question}"
    system    = ("You are a helpful assistant. Answer questions based on the provided document content. "
                 "Use markdown formatting where appropriate.")
    return Response(stream_with_context(stream_ai(prompt, system)),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/stream-image", methods=["POST"])
def stream_image():
    data       = request.json
    question   = data.get("question", "Describe this image in detail.")
    image_data = data.get("image_data", "")
    mime_type  = data.get("mime_type", "image/jpeg")
    if not image_data:
        def no_image():
            yield 'data: {"token": "No image data provided."}\n\n'
            yield "data: [DONE]\n\n"
        return Response(stream_with_context(no_image()), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    def gen_image_stream():
        try:
            stream = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{image_data}"}},
                        {"type": "text", "text": question}
                    ]
                }],
                max_tokens=1024,
                stream=True
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield f"data: {json.dumps({'token': delta})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'token': f'Error: {str(e)}'})}\n\n"
        yield "data: [DONE]\n\n"

    return Response(stream_with_context(gen_image_stream()),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


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