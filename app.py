from transformers import pipeline
import streamlit as st

# function part
# img2text
def img2text(url):
    image_to_text_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = image_to_text_model(url)[0]["generated_text"]
    return text

# text2story
def text2story(text):
    story_generator = pipeline("text-generation", model="gpt2")
    prompt = "Create a short children's story based on this image description: " + text + "\nStory:"
    story = story_generator(prompt, max_new_tokens=100, do_sample=True, temperature=0.8, top_p=0.9, repetition_penalty=1.2, no_repeat_ngram_size=3)
    story_text = story[0]["generated_text"]
    story_text = story_text.replace(prompt, "").strip()
    return story_text

# text2audio
def text2audio(story_text):
    audio_generator = pipeline("text-to-audio", model="Matthijs/mms-tts-eng")
    audio_data = audio_generator(story_text)
    return audio_data

# main part
st.set_page_config(page_title="Your Image to Audio Story", page_icon="🦜")
st.header("Turn Your Image to Audio Story")

uploaded_file = st.file_uploader("Select an Image...")

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()

    with open(uploaded_file.name, "wb") as file:
        file.write(bytes_data)

    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if st.button("Generate Story"):
        scenario = img2text(uploaded_file.name)
        story = text2story(scenario)
        st.write(story)

        audio_data = text2audio(story)
        audio_array = audio_data["audio"]
        sample_rate = audio_data["sampling_rate"]
        st.audio(audio_array, sample_rate=sample_rate)
