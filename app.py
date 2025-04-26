import os

os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
os.environ["STREAMLIT_SERVER_ENABLECORS"] = "false"
os.environ["STREAMLIT_SERVER_PORT"] = "10000"
os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import torch
import pytesseract
from PIL import Image
from transformers import pipeline, MarianMTModel, MarianTokenizer
from gtts import gTTS
import tempfile
import os

# ----------------------
# Title
st.title("🎥 Multilingual Assistive Vision Web App")

# ----------------------
# Language Selection
language_selection = st.selectbox(
    "🌎 Select Language:",
    ("English", "Hindi", "French", "Spanish", "German"),
)

lang_code_map = {
    "English": "en",
    "Hindi": "hi",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
}

selected_lang_code = lang_code_map[language_selection]

# Translation models
translation_models = {
    "hi": "Helsinki-NLP/opus-mt-en-hi",
    "fr": "Helsinki-NLP/opus-mt-en-fr",
    "es": "Helsinki-NLP/opus-mt-en-es",
    "de": "Helsinki-NLP/opus-mt-en-de",
}

# Load translation model if needed
if selected_lang_code != "en":
    trans_tokenizer = MarianTokenizer.from_pretrained(translation_models[selected_lang_code])
    trans_model = MarianMTModel.from_pretrained(translation_models[selected_lang_code])

# Image captioning model (smaller, works on Streamlit Cloud)
captioner = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

# Start/Stop controls
start_captioning = st.button("▶️ Start Captioning")
stop_captioning = st.button("⏹️ Stop Captioning")

# State flag
if "running" not in st.session_state:
    st.session_state.running = False

if start_captioning:
    st.session_state.running = True

if stop_captioning:
    st.session_state.running = False

# Display placeholder
output_placeholder = st.empty()

# ----------------------
# Video Processor

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_skip = 30  # capture every ~30 frames
        self.counter = 0

    def recv(self, frame):
        img = frame.to_image()

        if st.session_state.running and self.counter % self.frame_skip == 0:
            # Image Caption
            caption = captioner(img)[0]["generated_text"]

            # OCR Text
            ocr_text = pytesseract.image_to_string(img)

            combined_text = caption
            if ocr_text.strip():
                combined_text += f". Detected text: {ocr_text.strip()}"

            # Translation if needed
            final_output = combined_text
            if selected_lang_code != "en":
                inputs = trans_tokenizer(combined_text, return_tensors="pt", padding=True)
                translated = trans_model.generate(**inputs)
                final_output = trans_tokenizer.decode(translated[0], skip_special_tokens=True)

            # Update UI
            output_placeholder.markdown(f"### 📝 Output: {final_output}")

            # Audio Output
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                tts = gTTS(text=final_output, lang=selected_lang_code)
                temp_audio_path = fp.name
                tts.save(temp_audio_path)
            audio_placeholder.audio(temp_audio_path, format="audio/mp3")
            os.remove(temp_audio_path)

        self.counter += 1
        return av.VideoFrame.from_image(img)

# ----------------------
# WebRTC
audio_placeholder = st.empty()

webrtc_streamer(
    key="example",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

