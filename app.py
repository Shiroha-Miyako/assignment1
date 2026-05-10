# Program title: Storytelling App

# Import part
import streamlit as st
from transformers import pipeline

# Function part
def img2text(url):
    image_to_text_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text_model(url)[0]["generated_text"]
    return text

def text2story(text):
    story_pipe = pipeline("text-generation", model="pranavpsv/genre-story-generator-v2")
    prompt = "Write a short story for children based on: " + text
    story_results = story_pipe(prompt, max_length=120, min_length=60, do_sample=True, temperature=0.9, repetition_penalty=1.5, no_repeat_ngram_size=3)
    story_text = story_results[0]["generated_text"]
    return story_text

def text2audio(story_text):
    audio_pipe = pipeline("text-to-audio", model="Matthijs/mms-tts-eng")
    audio_data = audio_pipe(story_text)
    return audio_data

# Main part
st.set_page_config(page_title="Your Image to Audio Story", page_icon="🦜")
st.header("Turn Your Image to Audio Story")
uploaded_file = st.file_uploader("Select an Image...")

if uploaded_file is not None:
    # Save file locally
    bytes_data = uploaded_file.getvalue()
    with open(uploaded_file.name, "wb") as file:
        file.write(bytes_data)

    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Generate button
    if st.button("Generate Story"):
        # Stage 1: Image to Text
        st.text('Processing img2text...')
        scenario = img2text(uploaded_file.name)
        st.write(f"**Scenario:** {scenario}")

        # Stage 2: Text to Story
        st.text('Generating a story...')
        story = text2story(scenario)
        st.write(f"**Story:** {story}")

        # Stage 3: Story to Audio
        st.text('Generating audio data...')
        audio_data = text2audio(story)

        # Play Audio
        audio_array = audio_data["audio"]
        sample_rate = audio_data["sampling_rate"]
        st.audio(audio_array, sample_rate=sample_rate)
