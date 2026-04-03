from flask import Flask, request, jsonify, render_template
from groq import Groq

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
        answer = ask_ai(question)
        print("ANSWER:", answer)
        return jsonify({"answer": answer})
    except Exception as e:
        print("ERROR:", str(e))
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

if __name__ == "__main__":
    app.run(debug=True)