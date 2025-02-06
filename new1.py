from flask import Flask, render_template, request
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from openai import OpenAI
client = OpenAI()

app = Flask(__name__)

def load_knowledge_base(filepath):
    try:
        return pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(filepath, encoding='ISO-8859-1')

#knowledge_base = pd.read_csv('knowledge.csv')
knowledge_base = load_knowledge_base('knowledge.csv')
model_name = "gpt2"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def find_answer(question, knowledge_base):
    
    for idx, row in knowledge_base.iterrows():
        if row['question'].strip().lower() == question.strip().lower():
            return row['answer']
    return None

def generate_answer_with_openai(question):
    try:
        response = OpenAI.Completion.create(
            engine="text-davinci-003",  # Using Davinci for example
            prompt=question,
            max_tokens=150
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"Error: {e}"
    
def generate_answer(question):
    answer = find_answer(question, knowledge_base)
    if answer:
        return answer
    try:
        # Using OpenAI API for generating a fallback answer
        response = client.Completion.create(
            engine="text-davinci-003",  # Or other GPT models like gpt-3.5-turbo
            prompt=f"Q: {question}\nA:",
            max_tokens=150,
            temperature=0.7
        )
        generated_answer = response.choices[0].text.strip()
        return generated_answer
    except Exception as e:
        # Handle errors gracefully
        return "Sorry, I couldn't generate an answer at the moment. Please try again later."

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_question = request.form["question"]
        answer = generate_answer(user_question)
        return render_template("web.html", question=user_question, answer=answer)
    return render_template("web.html")

if __name__ == "__main__":
        app.run(debug=True)
