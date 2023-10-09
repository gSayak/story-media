import os
from dotenv import find_dotenv, load_dotenv
from bardapi import Bard
from transformers import pipeline, AutoTokenizer
import streamlit as st


load_dotenv(find_dotenv())
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")


def image2text(url):
    image_to_text = pipeline(
        "image-to-text", model="Salesforce/blip-image-captioning-base", tokenizer=AutoTokenizer.from_pretrained(
            "Salesforce/blip-image-captioning-base", legacy=False
        ))
    print(url)
    text = image_to_text(url)[0]["generated_text"]
    # print(text)
    return text


def generate_story(story):
    load_dotenv()
    # token = os.getenv("BARD_API_KEY")

    # bard = Bard(token=token)
    # text = image2text(story)

    headers = {
        "authorization": st.secrets["BARD_API_KEY"],
        "content-type": "application/json"
    }

    prompt = f"In not more than 30 words Generate a short story based on this caption: {story}"
    result = bard.get_answer(prompt, headers=headers)['content']
    # print(result)
    return result


def main():

    st.set_page_config(page_title="Story Generator", page_icon="📖")

    st.header("Turn image into story")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # print(uploaded_file)
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption='Uploaded Image.',
                 use_column_width=True)
        scenario = image2text(uploaded_file.name)
        story = generate_story(scenario)

        with st.expander("scenario"):
            st.write(scenario)
        with st.expander("story"):
            st.write(story)


if __name__ == "__main__":
    main()
