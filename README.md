YouTube Video Transcriber, Summarizer, and Translator

Overview

This project allows users to download a YouTube video, extract audio, transcribe speech into text, summarize the transcript, and translate it into multiple languages. The web interface is built using Streamlit.

Features

- Download YouTube Videos using yt_dlp.

- Extract and Convert Audio from videos using pydub.

- Transcribe Speech to Text using speech_recognition.

- Summarize the Transcript using facebook/bart-large-cnn.

- Translate the Summary into different languages using googletrans.

- Interactive Web Interface built with Streamlit.

Technologies Used

- Python

- yt_dlp (YouTube video downloading)

- pydub (Audio processing)

- speech_recognition (Speech-to-text conversion)

- transformers (BART model for summarization)

- googletrans (Translation API)

- Streamlit (Web application)

Installation

- Clone the repository:

git clone https://github.com/yourusername/YT-Transcriber-Summarizer.git

- Navigate to the project folder:

cd YT-Transcriber-Summarizer

- Install the dependencies:

pip install -r requirements.txt

- Run the Streamlit application:

streamlit run Model4.py

How It Works

- Enter the YouTube Video URL in the Streamlit app.

- The script downloads the video and extracts audio.

- The audio is transcribed into text using Google Speech Recognition.

- The text is summarized using the BART model.

- The summary can be translated into multiple languages.

The results are displayed in the web interface.

