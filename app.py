from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain import PromptTemplate, LLMChain, OpenAI
import os
import argparse
from bardapi import Bard

load_dotenv(find_dotenv())


def image2text(url):
    image_to_text = pipeline(
        "image-to-text", model="Salesforce/blip-image-captioning-base")

    text = image_to_text(url)[0]['generated_text']

    print(text)
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="bard-cli",
        description="Simple CLI to get prompt results from Google's BARD",
        epilog=":)",
    )
    parser.add_argument("--prompt")
    args = parser.parse_args()
    load_dotenv()
    token = os.getenv("BARD_API_KEY")

    bard = Bard(token=token)
    test = image2text("crawler.jpg")
    # result = bard.get_answer(test)
    # print(result)

    prompt = f"In not more than 30 words Generate a short story based on this image caption:\n{test}\n\nSTORY:"
    result = bard.get_answer(prompt)['content']
    print(result)
