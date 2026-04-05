import sqlite3
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from groq import Groq

def init_db():
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS messages
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  role TEXT,
                  content TEXT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

init_db()

app = Flask(__name__)

import os
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

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

@app.route("/chat", methods=["POST"])
def chat():
    question = request.json.get("question")
    try:
        answer = ask_ai(question, "You are a helpful AI assistant. Answer any question clearly.")
        
        # Save to database
        conn = sqlite3.connect('chat_history.db')
        c = conn.cursor()
        c.execute("INSERT INTO messages (role, content) VALUES (?, ?)", ("user", question))
        c.execute("INSERT INTO messages (role, content) VALUES (?, ?)", ("ai", answer))
        conn.commit()
        conn.close()
        
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
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute("SELECT role, content FROM messages ORDER BY id DESC LIMIT 20")
    messages = [{"role": row[0], "content": row[1]} for row in c.fetchall()]
    conn.close()
    return jsonify({"messages": messages[::-1]})

@app.route("/clear", methods=["POST"])
def clear():
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute("DELETE FROM messages")
    conn.commit()
    conn.close()
    return jsonify({"status": "cleared"})

if __name__ == "__main__":
    app.run(debug=True)