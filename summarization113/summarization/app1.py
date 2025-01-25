import os
import cv2
import whisper
from yt_dlp import YoutubeDL
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
from pymongo import MongoClient
from datetime import datetime
import logging
from googletrans import Translator
from flask import Flask, render_template, request, jsonify, send_file

app = Flask(__name__)

# Load Whisper model and BART model once at the start
whisper_model = whisper.load_model("base")
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bart_model.to(device)

# MongoDB connection setup
mongo_client = MongoClient("mongodb+srv://shyamala:wyj.hdNrX8hGDub@cluster0.0f6zw.mongodb.net/")
db = mongo_client["video_database"]
collection = db["transcripts"]
def generate_summary(text):
    inputs = bart_tokenizer.encode("summarize: " + text, return_tensors='pt', max_length=1024, truncation=True).to(device)
    summary_ids = bart_model.generate(inputs, max_length=600, min_length=150, length_penalty=2.0, num_beams=4, early_stopping=True)
    return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/get_dynamic_summary', methods=['POST'])
def get_dynamic_summary():
    data = request.json
    url = data.get('url')
    start_time = int(data.get('startTime'))
    end_time = int(data.get('endTime'))

    # Retrieve the transcript from the database
    video_record = collection.find_one({"url": url})
    if video_record:
        transcript = video_record['transcript']
        
        # Extract the portion of the transcript based on start and end time
        # Implement logic to extract the relevant segment from the video
        summary = generate_summary(transcript[start_time:end_time])
        return jsonify({'summary': summary})
    else:
        return jsonify({'error': 'Video URL not found in the database.'}), 404

@app.route('/dynamic_summary')
def dynamic_summary():
    return render_template('dynamic.html')

@app.route('/process', methods=['POST'])
def process_video():
    data = request.json
    video_url = data.get('url')

    existing_record = collection.find_one({"url": video_url})
    if existing_record:
        return jsonify({'summary': existing_record['summary']})

    video_id, transcript = download_video_and_transcribe(video_url)
    if transcript:
        summary = generate_summary(transcript)
        collection.insert_one({
            "url": video_url,
            "transcript": transcript,
            "summary": summary,
            "date_processed": datetime.utcnow()
        })
        return jsonify({'summary': summary})
    else:
        return jsonify({'error': 'No transcript was generated.'})

@app.route('/translate', methods=['POST'])
def translate_summary():
    data = request.json
    summary = data.get('summary')
    target_language = data.get('target_language')

    if not summary:
        return jsonify({'error': 'No summary provided'}), 400

    if not target_language:
        return jsonify({'error': 'Target language not provided'}), 400

    try:
        translator = Translator()
        translated_summary = translator.translate(summary, dest=target_language).text
        return jsonify({'translated_summary': translated_summary})

    except Exception as e:
        logging.error(f"Translation error: {e}")
        return jsonify({'error': 'Translation failed'}), 500

@app.route('/supported-languages', methods=['GET'])
def get_languages():
    languages = {
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "zh": "Chinese",
        "hi": "Hindi",
        "ja": "Japanese",
        "ko": "Korean",
        "pt": "Portuguese",
        "ar": "Arabic",
        "ru": "Russian"
    }
    return jsonify({'languages': languages})

@app.route('/check_url', methods=['POST'])
def check_url():
    data = request.get_json()
    url = data.get('url')

    existing_record = collection.find_one({"url": url})
    if existing_record:
        return jsonify({"exists": True})
    else:
        video_id, transcript = download_video_and_transcribe(url)
        if transcript:
            summary = generate_summary(transcript)
            collection.insert_one({
                "url": url,
                "transcript": transcript,
                "summary": summary,
                "date_processed": datetime.utcnow()
            })
            return jsonify({"exists": False, "message": "New video processed and stored in the database."})
        else:
            return jsonify({"error": "Error processing the video. Transcript not generated."})

@app.route('/static-demo')
def static_demo():
    return send_file('static/static.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)
