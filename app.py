import streamlit as st
import cv2
import torch
from PIL import Image
import numpy as np
import pytesseract
from transformers import BlipProcessor, BlipForConditionalGeneration, MarianMTModel, MarianTokenizer
from gtts import gTTS
import tempfile
import time
import base64

# --- Title
st.title("🧠 Multilingual Assistive Vision")

# --- Sidebar for language selection
st.sidebar.header("Settings")
selected_language = st.sidebar.selectbox(
    "Choose output language",
    ("English", "Hindi", "French", "Spanish", "German")
)

# Start/Stop button
start_captioning = st.sidebar.button("▶️ Start Captioning")
stop_captioning = st.sidebar.button("⏹️ Stop Captioning")

# Language map
lang_code_map = {
    "English": "en",
    "Hindi": "hi",
    "French": "fr",
    "Spanish": "es",
    "German": "de"
}

translation_model_map = {
    "hi": "Helsinki-NLP/opus-mt-en-hi",
    "fr": "Helsinki-NLP/opus-mt-en-fr",
    "es": "Helsinki-NLP/opus-mt-en-es",
    "de": "Helsinki-NLP/opus-mt-en-de"
}

# --- Load models
@st.cache_resource
def load_models():
    device = torch.device("cpu")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    return processor, caption_model, device

processor, caption_model, device = load_models()

def load_translation_model(lang_code):
    if lang_code == "en":
        return None, None  # No translation needed
    model_name = translation_model_map.get(lang_code)
    if model_name:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name).to(device)
        return tokenizer, model
    return None, None

# Load translation model based on selected language
target_lang_code = lang_code_map[selected_language]
trans_tokenizer, trans_model = load_translation_model(target_lang_code)

# --- Webcam
frame_window = st.image([])

if start_captioning:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("🚫 Cannot access webcam.")
        st.stop()

    st.success("🚀 Capturing started... Click ⏹️ Stop Captioning to stop.")

    frame_skip = 10
    frame_count = 0

    stop_signal = False

    # Create a session state for stopping
    if "stop" not in st.session_state:
        st.session_state.stop = False

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("⚠️ Failed to capture frame.")
            break

        # Show webcam frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame_rgb, channels="RGB")

        frame_count += 1

        if frame_count % frame_skip == 0:
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)

            # 1. Caption the image
            inputs = processor(pil_image, return_tensors="pt").to(device)
            with torch.no_grad():
                out = caption_model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)

            # 2. OCR text detection
            ocr_text = pytesseract.image_to_string(pil_image, lang="eng").strip()

            # 3. Combine caption and OCR
            combined_text = caption
            if ocr_text:
                combined_text += f". Detected text: {ocr_text}"

            # 4. Translate if needed
            if target_lang_code != "en" and trans_model is not None:
                trans_inputs = trans_tokenizer(combined_text, return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    translated = trans_model.generate(**trans_inputs)
                translated_text = trans_tokenizer.decode(translated[0], skip_special_tokens=True)
            else:
                translated_text = combined_text

            # 5. Display translated text
            st.subheader("📝 Description:")
            st.write(translated_text)

            # 6. Speak translated text automatically
            tts = gTTS(text=translated_text, lang=target_lang_code)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                tts.save(fp.name)
                audio_file = fp.name

            # Read audio file and play automatically
            with open(audio_file, "rb") as audio:
                audio_bytes = audio.read()
                b64 = base64.b64encode(audio_bytes).decode()

                md = f"""
                <audio autoplay="true" controls>
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                </audio>
                """
                st.markdown(md, unsafe_allow_html=True)

            # Sleep before next caption
            time.sleep(5)

        # Check for stop button
        if stop_captioning:
            st.warning("🛑 Captioning stopped manually.")
            break

    # Release camera
    cap.release()
