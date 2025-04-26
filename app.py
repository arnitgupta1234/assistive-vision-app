import streamlit as st
import cv2
import torch
from PIL import Image
import pytesseract
from transformers import BlipProcessor, BlipForConditionalGeneration, MarianMTModel, MarianTokenizer
from gtts import gTTS
import os
import tempfile
import time

# Setup
device = torch.device("cpu")

@st.cache_resource
def load_models():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    return processor, caption_model

@st.cache_resource
def load_translator(lang_code):
    lang_map = {
        "fr": "Helsinki-NLP/opus-mt-en-fr",
        "es": "Helsinki-NLP/opus-mt-en-es",
        "hi": "Helsinki-NLP/opus-mt-en-hi",
        "de": "Helsinki-NLP/opus-mt-en-de"
    }
    model_name = lang_map[lang_code]
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device)
    return tokenizer, model

# UI
st.title("🧠🔊 Continuous Assistive Vision")
st.markdown("Image Captioning + OCR + Multilingual TTS")

target_lang = st.selectbox("🌐 Choose output language", ["hi", "fr", "es", "de"])
start_btn = st.button("📷 Start Assistive Camera")

processor, caption_model = load_models()
trans_tokenizer, trans_model = load_translator(target_lang)

# Capture + process
if start_btn:
    cap = cv2.VideoCapture(0)
    frame_area = st.empty()
    text_area = st.empty()
    stop_btn = st.button("🛑 Stop")

    frame_count = 0
    skip_frames = 10
    last_output = ""

    while cap.isOpened() and not stop_btn:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Camera frame unavailable.")
            break

        # Show frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_area.image(frame_rgb, channels="RGB")

        if frame_count % skip_frames == 0:
            pil_image = Image.fromarray(frame_rgb)

            # Caption
            inputs = processor(pil_image, return_tensors="pt").to(device)
            with torch.no_grad():
                output = caption_model.generate(**inputs)
            caption = processor.decode(output[0], skip_special_tokens=True)

            # OCR
            ocr_text = pytesseract.image_to_string(pil_image, lang="eng").strip()
            combined = caption + (f". Detected text: {ocr_text}" if ocr_text else "")

            # Translate
            trans_inputs = trans_tokenizer(combined, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                translated = trans_model.generate(**trans_inputs)
            translated_text = trans_tokenizer.decode(translated[0], skip_special_tokens=True)

            # Speak if new
            if translated_text != last_output:
                text_area.markdown(f"**📝 {translated_text}**")
                tts = gTTS(text=translated_text, lang=target_lang)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                    path = fp.name
                tts.save(path)
                os.system(f"start {path}")  # Windows; replace with afplay/mpg123 for Mac/Linux
                last_output = translated_text

        frame_count += 1
        time.sleep(0.1)  # non-blocking pause

    cap.release()
    st.success("📷 Camera stopped.")
