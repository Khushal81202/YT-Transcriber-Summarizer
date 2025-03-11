import yt_dlp
import os
from pydub import AudioSegment
import speech_recognition as sr
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration
import re
import sys
import io
import streamlit as st
from googletrans import Translator  # Import the Translator

# Load the summarization model and tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
translator = Translator()  # Initialize the Translatorṇ

def download_video(url, download_path="downloads"):
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': f'{download_path}/%(title)s.%(ext)s',
        'noplaylist': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        video_file = ydl.prepare_filename(info_dict)
    return video_file

def extract_audio(video_path, audio_path):
    audio = AudioSegment.from_file(video_path)
    audio.export(audio_path, format="mp3")

def convert_to_wav(mp3_path, wav_path):
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")

def clean_text(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)

def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    audio = AudioSegment.from_wav(audio_path)
    chunk_length_ms = 60000  # 1 minute
    chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
    full_transcript = ""
    for i, chunk in enumerate(chunks):
        chunk.export("temp_chunk.wav", format="wav")
        with sr.AudioFile("temp_chunk.wav") as source:
            audio_data = recognizer.record(source)
            try:
                transcript = recognizer.recognize_google(audio_data)
                full_transcript += transcript + " "
            except sr.UnknownValueError:
                print(f"Could not understand audio for chunk {i+1}")
            except sr.RequestError as e:
                print(f"Request error for chunk {i+1}; {e}")
    return full_transcript

def summarize_in_chunks(text, chunk_size=1024):
    text = clean_text(text)
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    return " ".join(summaries)

def translate_text(text, target_language):
    translation = translator.translate(text, dest=target_language)
    return translation.text

def main():
    st.title("YouTube Video Transcriber, Summarizer, and Translator")
    video_url = st.text_input("Enter the YouTube video URL")

    if video_url:
        st.write("Processing video...")
        download_path = "downloads"
        os.makedirs(download_path, exist_ok=True)

        video_file = download_video(video_url, download_path)
        st.write(f"Downloaded video file: {video_file}")

        audio_file_mp3 = os.path.join("D:\B.tech mini project\downloads", os.path.splitext(os.path.basename(video_file))[0] + ".mp3")
        extract_audio(video_file, audio_file_mp3)

        audio_file_wav = os.path.splitext(audio_file_mp3)[0] + ".wav"
        convert_to_wav(audio_file_mp3, audio_file_wav)

        transcript = transcribe_audio(audio_file_wav)
        st.subheader("Full Transcript:")
        st.write(transcript)

        summarized_transcript = summarize_in_chunks(transcript)
        st.subheader("Summarized Transcript:")
        st.write(summarized_transcript)

        # Language selection for translation
        languages = {'Spanish': 'es', 'French': 'fr', 'German': 'de', 'Chinese': 'zh-cn', 'Hindi': 'hi'}
        selected_language = st.selectbox("Select a language for translation", list(languages.keys()))

        if st.button("Translate Summary"):
            translated_text = translate_text(summarized_transcript, languages[selected_language])
            st.subheader(f"Translated Summary in {selected_language}:")
            st.write(translated_text)

if __name__ == "__main__":
    main()
