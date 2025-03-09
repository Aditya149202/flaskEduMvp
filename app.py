# app.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import os
from werkzeug.utils import secure_filename
from extract import extract_text_from_pdf

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the flan-t5-small model and tokenizer
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    text = data.get("text", "")
    difficulty = data.get("difficulty", "easy")
    
    # Build a prompt that includes the difficulty parameter.
    # You can adjust the format to best suit your fine-tuning.
    prompt = f"Based on the following passage, generate a question that tests a student's understanding of its main concept.Difficulty:${difficulty} Text:${text}"

    
    inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=100, num_beams=4,num_return_sequences=3, early_stopping=True)
    generated_questions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    
    questions_with_options = []
    for question in generated_questions:
        options_prompt = (
            f"Generate four multiple-choice answer options for the following question, separated by commas, where one option is correct: {question}"
        )
        options_inputs = tokenizer.encode(options_prompt, return_tensors="pt", max_length=512, truncation=True)
        options_outputs = model.generate(options_inputs, max_length=80, num_beams=4, num_return_sequences=1, early_stopping=True)
        print(options_outputs)
        options_text = tokenizer.decode(options_outputs[0], skip_special_tokens=True)
        # Clean up the options_text if it contains an unwanted prefix.
        options_text = options_text.replace("options:", "").strip()
        options = [option.strip() for option in options_text.split(",") if option.strip()]
        questions_with_options.append({"question": question, "options": options})

        
    return jsonify({"questions": questions_with_options})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        extracted_text = extract_text_from_pdf(file_path)
        return jsonify({"text": extracted_text})

if __name__ == '__main__':
    app.run(debug=True,port=5000)
