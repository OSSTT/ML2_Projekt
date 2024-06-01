from flask import Flask, request, jsonify, send_from_directory, session
import base64
import requests
from prediction import predict_image
import os
from werkzeug.utils import secure_filename
import json
from datetime import datetime
from flask_session import Session
from pathlib import Path
import openai

app = Flask(__name__, static_url_path='', static_folder='frontend')

# Flask-Session konfigurieren
app.config['SESSION_TYPE'] = 'filesystem'
app.secret_key = 'supersecretkey'
Session(app)

api_key = "sk-proj-6nAaNjroNYSvmIyAJgwoT3BlbkFJaxl62nynTAQLIseJBoYG"
openai.api_key = api_key

def get_openai_response(messages):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": "gpt-4o",
        "messages": messages,
        "max_tokens": 300
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()

def save_to_results_json(data):
    json_filepath = os.path.join('results', 'results_api.json')
    os.makedirs('results', exist_ok=True)
    
    if os.path.exists(json_filepath):
        with open(json_filepath, 'r') as json_file:
            existing_data = json.load(json_file)
    else:
        existing_data = []

    existing_data.append(data)
    
    with open(json_filepath, 'w') as json_file:
        json.dump(existing_data, json_file, indent=4)

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/text', methods=['POST'])
def handle_text():
    data = request.json
    user_message = data.get('text', '')

    # Unterhaltungshistorie aus der Sitzung abrufen oder initialisieren
    conversation_history = session.get('conversation_history', [
        {"role": "system", "content": "You are an expert in art, especially when it comes to artworks."}
    ])

    # Benutzer-Nachricht zur Historie hinzufügen
    conversation_history.append({"role": "user", "content": user_message})

    # Antwort von OpenAI abrufen
    openai_response = get_openai_response(conversation_history)
    reply = openai_response['choices'][0]['message']['content']

    # Antwort des Assistenten zur Historie hinzufügen
    conversation_history.append({"role": "assistant", "content": reply})

    # Historie in der Sitzung speichern
    session['conversation_history'] = conversation_history

    # Antwort in results_api.json speichern
    save_to_results_json({
        "type": "text",
        "user_message": user_message,
        "reply": reply,
        "timestamp": datetime.now().isoformat()
    })

    return jsonify({"reply": reply})

@app.route('/api/image', methods=['POST'])
def handle_image():
    data = request.json
    image_data = data.get('image', '')

    if not image_data.startswith('data:image/jpeg;base64,'):
        return jsonify({"error": "Invalid image data"}), 400

    # Extract base64 encoded image data
    base64_image = image_data.split(',')[1]

    # Lesen des letzten Eintrags aus der JSON-Datei
    json_filepath = os.path.join('results', 'prediction_results.json')
    if os.path.exists(json_filepath):
        with open(json_filepath, 'r') as json_file:
            data = json.load(json_file)
            if data:
                last_result = data[-1]["predicted_class"]
            else:
                last_result = "unknown"
    else:
        last_result = "unknown"

    prompt = f'This image was drawn from "{last_result}". What’s the name of this image and what is happening in this image?'

    # Unterhaltungshistorie aus der Sitzung abrufen oder initialisieren
    conversation_history = session.get('conversation_history', [
        {"role": "system", "content": "You are an expert in art, especially when it comes to artworks."}
    ])

    # Benutzer-Nachricht mit Bild zur Historie hinzufügen
    conversation_history.append(
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}", "detail": "high"}}
        ]}
    )

    # Antwort von OpenAI abrufen
    openai_response = get_openai_response(conversation_history)
    reply = openai_response['choices'][0]['message']['content']

    # Antwort des Assistenten zur Historie hinzufügen
    conversation_history.append({"role": "assistant", "content": reply})

    # Historie in der Sitzung speichern
    session['conversation_history'] = conversation_history

    # Antwort in results_api.json speichern
    save_to_results_json({
        "type": "image",
        "prompt": prompt,
        "reply": reply,
        "timestamp": datetime.now().isoformat()
    })

    return jsonify({"reply": reply})

@app.route('/api/predict', methods=['POST'])
def handle_prediction():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)
        predicted_class = predict_image(file_path)
        os.remove(file_path)
        
        return jsonify({"prediction": predicted_class})

@app.route('/api/last_prediction', methods=['GET'])
def get_last_prediction():
    json_filepath = os.path.join('results', 'prediction_results.json')
    
    if os.path.exists(json_filepath):
        with open(json_filepath, 'r') as json_file:
            data = json.load(json_file)
            if data:
                last_result = data[-1]
                return jsonify({"predicted_class": last_result["predicted_class"]})
            else:
                return jsonify({"error": "No predictions found"}), 404
    else:
        return jsonify({"error": "No prediction results file found"}), 404

@app.route('/api/speech', methods=['GET'])
def create_speech():
    # Setze den Pfad zum "results"-Ordner und zur Ausgabedatei
    results_folder = Path(__file__).parent / "results"
    speech_file_path = results_folder / "speech.mp3"
    json_filepath = results_folder / "results_api.json"

    # Erstelle den "results"-Ordner, falls er noch nicht existiert
    results_folder.mkdir(parents=True, exist_ok=True)

    # Lesen des letzten Eintrags aus der JSON-Datei
    if os.path.exists(json_filepath):
        with open(json_filepath, 'r') as json_file:
            data = json.load(json_file)
            if data:
                last_reply = data[-1]["reply"]
            else:
                return jsonify({"error": "No replies found"}), 404
    else:
        return jsonify({"error": "results_api.json file not found"}), 404

    # Erstelle die Text-to-Speech-Anfrage
    response = openai.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=last_reply
    )

    # Speichere die Antwort im "results"-Ordner
    response.stream_to_file(speech_file_path)

    return jsonify({"message": "Speech file created successfully", "file_path": str(speech_file_path)})

@app.route('/api/get_speech', methods=['GET'])
def get_speech():
    results_folder = Path(__file__).parent / "results"
    speech_file_path = results_folder / "speech.mp3"

    if speech_file_path.exists():
        return send_from_directory(directory=results_folder, path="speech.mp3", as_attachment=True)
    else:
        return jsonify({"error": "speech.mp3 file not found"}), 404

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)
