from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from groq import Groq
import os
import gc
from supabase import create_client
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Create a folder to store uploaded PDFs
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# This will store your searchable document data in memory
vector_db = None

app = Flask(__name__)
CORS(app)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
db = create_client(supabase_url, supabase_key)

def ask_ai(prompt, system="You are a helpful study assistant."):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    global vector_db
    # Check if a file was actually sent
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # 1. Securely save the file to the 'uploads' folder
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        # 2. Load the PDF content
        loader = PyPDFLoader(filepath)
        pages = loader.load()

        # 3. Split the text into smaller chunks (better for AI context)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(pages)

        # 4. Convert text chunks into searchable vectors (embeddings)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2", 
            model_kwargs={'device': 'cpu'}
        )
        gc.collect()
        
        # 5. Store them in the FAISS vector database
        vector_db = FAISS.from_documents(docs, embeddings)

        return jsonify({"message": f"Successfully indexed {filename}!"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/chat", methods=["POST"])
def chat():
    question = request.json.get("question")
    try:
        answer = ask_ai(question, "You are a helpful AI assistant. Answer any question clearly.")
        db.table("messages").insert({"role": "user", "content": question}).execute()
        db.table("messages").insert({"role": "ai", "content": answer}).execute()
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"answer": f"Error: {str(e)}"})
    
@app.route("/summarize", methods=["POST"])
def summarize():
    notes = request.json.get("notes")
    answer = ask_ai(notes, "Summarize these notes into: 1) A short overview, 2) Key concepts as bullet points, 3) Important terms to remember.")
    return jsonify({"answer": answer})

@app.route("/flashcards", methods=["POST"])
def flashcards():
    topic = request.json.get("topic")
    count = request.json.get("count", 5)
    prompt = f"Create {count} flashcards about: {topic}. Reply ONLY with a JSON array like: [{{\"q\":\"question\",\"a\":\"answer\"}}]. No extra text."
    answer = ask_ai(prompt)
    return jsonify({"answer": answer})

@app.route("/studyplan", methods=["POST"])
def studyplan():
    subject = request.json.get("subject")
    days = request.json.get("days", 7)
    goal = request.json.get("goal", "general mastery")
    prompt = f"Subject: {subject}\nDays: {days}\nGoal: {goal}\nCreate a day-by-day study plan."
    answer = ask_ai(prompt, "You are a study planner. Create a clear, realistic day-by-day study plan.")
    return jsonify({"answer": answer})

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