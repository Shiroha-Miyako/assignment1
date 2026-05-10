from transformers import pipeline
from PIL import Image
from gtts import gTTS
from io import BytesIO
import streamlit as st

# function part
# img2text
def img2text(url):
    image_to_text_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text_model(url)[0]["generated_text"]
    return text

# text2story
def text2story(text):
    story_generator = pipeline("text2text-generation", model="google/flan-t5-base")
    prompt = "Write a short, simple, happy story for children aged 3 to 10 in 50 to 100 words based on this image: " + text
    story = story_generator(prompt, max_new_tokens=120, repetition_penalty=1.5)
    story_text = story[0]["generated_text"]
    return story_text

# text2audio
def text2audio(story_text):
    tts = gTTS(text=story_text, lang="en")
    audio_data = BytesIO()
    tts.write_to_fp(audio_data)
    audio_data.seek(0)
    return audio_data

# main part
st.title("AI Storytelling Application")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image)

    if st.button("Generate Story"):
        image_text = img2text(image)

        st.write(image_text)

        story_text = text2story(image_text)

        st.write(story_text)

        audio_data = text2audio(story_text)

        st.audio(audio_data, format="audio/mp3")
