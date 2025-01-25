**Video Summarization and Translation Tool**

This project provides a Flask-based web application for video summarization, dynamic summarization by time ranges, and multilingual translation of generated summaries. The application leverages Whisper for audio transcription, BART for summarization, and integrates MongoDB for efficient data storage and retrieval.

**Features**

1. Video Summarization
Downloads videos from a given URL using yt-dlp.
Transcribes the audio using OpenAI's Whisper model.
Generates concise summaries of the transcribed text using the BART model.
2. Dynamic Summary
Extracts and summarizes specific sections of the video transcript based on user-defined time ranges.
3. Translation
Translates the summary into various languages using Google Translate.
Supports popular languages like Spanish, French, German, Chinese, Hindi, and more.
4. Regeneration of Summaries
Allows users to regenerate summaries with a fixed length to match specific requirements.
5. Database Integration
Stores video URLs, transcripts, and summaries in MongoDB for efficient lookup and reuse.
6. Interactive UI
User-friendly interface built using Flask and HTML templates.
Includes static demos for showcasing functionality.

**Technologies Used**

Backend Framework: Flask

Transcription Model: Whisper (OpenAI)

Summarization Model: Facebook BART (Large CNN)

Database: MongoDB (Cloud-based with MongoDB Atlas)

Video Downloader: yt-dlp

Translation API: Google Translator

Frontend: HTML, CSS, JavaScript (for templates)

**Setup Instructions**

**Prerequisites**

Python 3.8 or higher

MongoDB instance (local or cloud)

CUDA-enabled GPU (optional but recommended for better performance)

Required Python packages (see requirements.txt)

**Usage**

Access the application in your browser.

Enter the video URL to summarize.

Specify time ranges for dynamic summaries if needed.

Translate summaries into supported languages.

**File Structure**

video-summarization-tool/

├── static/               # Static assets (HTML, CSS, JS)

├── templates/            # Flask templates (HTML files)

├── app.py                # Main Flask application

├── requirements.txt      # Python dependencies

├── README.md             # Project documentation

└── downloads/            # Folder for temporary video downloads


**Supported Languages**

The app supports the following languages for translation:

Spanish (es)

French (fr)

German (de)

Italian (it)

Chinese (zh)

Hindi (hi)

Japanese (ja)

Korean (ko)

Portuguese (pt)

Arabic (ar)

Russian (ru)

**Future Enhancements**

Add support for other video formats.

Enhance time-based extraction for dynamic summaries.

Improve translation accuracy with a custom translation model.

**Contributing**

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

