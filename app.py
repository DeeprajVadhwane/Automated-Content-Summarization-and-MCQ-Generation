import streamlit as st
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer, BartTokenizer, BartForConditionalGeneration, GPT2Tokenizer, GPT2LMHeadModel
import moviepy.editor as mp
import librosa
import tempfile
import os

st.title("AI-Powered MCQ Generator from Video")

# File uploader for video
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "move"])


if uploaded_file is not None:
    # Save the uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name + ".mp4"

    # Extract audio from the video using moviepy
    st.write("Extracting audio from the video...")
    video = mp.VideoFileClip(video_path)
    audio_path = tempfile.mktemp(suffix=".wav")
    video.audio.write_audiofile(audio_path)

    st.write("Audio extracted successfully!")

def transcribe_audio(audio_path):
    st.write("Transcribing audio to text...")

    # Load Wav2Vec2 model and tokenizer
    tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    # Load audio file using librosa
    audio_input, _ = librosa.load(audio_path, sr=16000)
    input_values = tokenizer(audio_input, return_tensors="pt").input_values

    # Perform transcription
    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.decode(predicted_ids[0])
    return transcription

if uploaded_file is not None:
    transcription = transcribe_audio(audio_path)
    st.write("Transcription:", transcription)

def summarize_text(text):
    st.write("Summarizing transcription...")

    # Load BART model and tokenizer
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

    inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

if uploaded_file is not None:
    summary = summarize_text(transcription)
    st.write("Summary:", summary)

def generate_question(summary):
    st.write("Generating a multiple-choice question...")

    # Load GPT-2 model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Format the input for the model
    prompt = f"Generate a multiple-choice question based on the following summary: {summary}\n"
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    # Generate the question
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True)
    question = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return question

if uploaded_file is not None:
    question = generate_question(summary)
    st.write("Generated MCQ:", question)
