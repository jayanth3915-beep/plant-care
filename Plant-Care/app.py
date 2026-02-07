import os
import json
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from groq import Groq

app = Flask(__name__)
app.secret_key = "f3a7b2c1d9e8f4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9" # Required for sessions

# --- CONFIGURATION ---
GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Local AI Model
MODEL_PATH = os.path.join(BASE_DIR, "mobilenetv2_best.keras")
CLASS_PATH = os.path.join(BASE_DIR, "class_indices.json")

model = load_model(MODEL_PATH)
with open(CLASS_PATH, 'r') as f:
    class_indices = json.load(f)
class_names = {v: k for k, v in class_indices.items()}

def get_groq_advice(disease_name, lang):
    try:
        prompt = (f"The plant disease is {disease_name}. Provide a short cure and prevention tips "
                  f"in 3-4 bullet points. You MUST respond ONLY in the {lang} language.")
        
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a professional plant pathologist."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Advice unavailable. Error: {str(e)}"

# --- ROUTES ---

@app.route('/')
def login():
    # If already logged in, go to uploader
    if 'user' in session:
        return redirect(url_for('uploader'))
    return render_template('login.html')

@app.route('/login_action', methods=['POST'])
def login_action():
    email = request.form.get('email')
    if email:
        session['user'] = email
        return redirect(url_for('uploader'))
    return redirect(url_for('login'))

@app.route('/uploader')
def uploader():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return redirect(url_for('login'))
        
    selected_lang = request.form.get('language', 'English')
    file = request.files['file']
    
    if file:
        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        img = image.load_img(filepath, target_size=(64, 64))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        preds = model.predict(img_array)
        top_idx = np.argmax(preds[0])
        disease_display = class_names[top_idx].replace("___", " - ").replace("_", " ")
        confidence = round(float(preds[0][top_idx]) * 100, 2)

        advice = get_groq_advice(disease_display, selected_lang)

        return render_template('index.html', 
                               prediction=disease_display, 
                               confidence=confidence, 
                               advice=advice, 
                               selected_lang=selected_lang,
                               image_path=f"static/uploads/{filename}")

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)