import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import numpy as np
import torch
from PIL import Image
import pytesseract
from transformers import BlipProcessor, BlipForConditionalGeneration, MarianMTModel, MarianTokenizer
from gtts import gTTS
import base64
from io import BytesIO

# Load models outside to avoid reloading every frame
device = torch.device("cpu")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Language selection
st.title("🌐 Multilingual Assistive Vision Web App")
target_lang = st.selectbox("Choose output language:", ["English", "Hindi", "French", "Spanish", "German"])

lang_code_map = {
    "English": "en",
    "Hindi": "hi",
    "French": "fr",
    "Spanish": "es",
    "German": "de"
}
target_code = lang_code_map[target_lang]

translation_models = {
    "hi": "Helsinki-NLP/opus-mt-en-hi",
    "fr": "Helsinki-NLP/opus-mt-en-fr",
    "es": "Helsinki-NLP/opus-mt-en-es",
    "de": "Helsinki-NLP/opus-mt-en-de"
}

if target_code != "en":
    trans_tokenizer = MarianTokenizer.from_pretrained(translation_models[target_code])
    trans_model = MarianMTModel.from_pretrained(translation_models[target_code]).to(device)

# For state management
if "run_captioning" not in st.session_state:
    st.session_state.run_captioning = True

# Stop button
if st.button("Stop Captioning"):
    st.session_state.run_captioning = False

# Function to convert text to speech and play
def speak_text(text, lang="en"):
    tts = gTTS(text=text, lang=lang)
    buf = BytesIO()
    tts.write_to_fp(buf)
    audio_bytes = buf.getvalue()
    st.audio(audio_bytes, format="audio/mp3")

# Define video processor
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        if st.session_state.run_captioning:
            # Process image
            pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            # Caption
            inputs = processor(pil_image, return_tensors="pt").to(device)
            with torch.no_grad():
                out = caption_model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)

            # OCR
            ocr_text = pytesseract.image_to_string(pil_image, lang="eng").strip()

            combined_text = caption
            if ocr_text:
                combined_text += f". Detected text: {ocr_text}"

            # Translation
            if target_code != "en":
                trans_inputs = trans_tokenizer(combined_text, return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    translated = trans_model.generate(**trans_inputs)
                final_text = trans_tokenizer.decode(translated[0], skip_special_tokens=True)
            else:
                final_text = combined_text

            st.session_state.latest_caption = final_text

            # Speak automatically
            speak_text(final_text, lang=target_code)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Stream video
webrtc_streamer(
    key="example",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Display latest caption
if "latest_caption" in st.session_state:
    st.subheader("📝 Latest Caption + OCR Output")
    st.write(st.session_state.latest_caption)
