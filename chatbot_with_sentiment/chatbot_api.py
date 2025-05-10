from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Dict
from openai import OpenAI
import requests
import time

app = FastAPI()

# === Chatbot API Setup ===
HUGGINGFACE_DEDICATED_ENDPOINT = "https://h80ls72bh8kqg12h.us-east-1.aws.endpoints.huggingface.cloud/v1/"
HUGGINGFACE_ACCESS_TOKEN = "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"
bos_token = "<s>"
eos_token = "</s>"

client = OpenAI(
    base_url=HUGGINGFACE_DEDICATED_ENDPOINT,
    api_key=HUGGINGFACE_ACCESS_TOKEN,
)

# === Sentiment Classifier Setup ===
SENTIMENT_API_URL = "https://fok73vtkey70bt77.us-east-1.aws.endpoints.huggingface.cloud"
SENTIMENT_HEADERS = {
    "Authorization": "Bearer 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",
    "Content-Type": "application/json"
}
labels = {
    "LABEL_0": "Normal",
    "LABEL_1": "Depression",
    "LABEL_2": "Suicidal",
    "LABEL_3": "Anxiety",
    "LABEL_4": "Stress",
    "LABEL_5": "Bipolar",
    "LABEL_6": "Personality disorder"
}

chat_history: List[Dict[str, str]] = []

class UserMessage(BaseModel):
    message: str

def classify_text(text):
    response = requests.post(SENTIMENT_API_URL, headers=SENTIMENT_HEADERS, json={"inputs": text})
    output = response.json()
    if isinstance(output, list) and len(output) > 0:
        predictions = output[0]
        label_id = predictions["label"]
        label = labels.get(label_id, "Unknown")
        score = predictions["score"]
        return label, score
    return "Unknown", None

def build_prompt(chat_history):
    prompt = bos_token
    prompt += (
        "You are a compassionate therapist providing empathetic and supportive responses. "
        "Engage with the user in a caring, thoughtful manner while offering gentle guidance.\n"
        "Always consider the user's emotional sentiment (e.g., Depression, Stress, etc.) when crafting your replies to better address their state of mind.\n"
    )
    for message in chat_history:
        if message["from"] == "human":
            prompt += f"User: {message['value']}\n"
        elif message["from"] == "gpt":
            prompt += f"Therapist: {message['value']}\n"
    prompt += eos_token + "\nTherapist: "
    return prompt

@app.post("/chat")
def chat(user_msg: UserMessage):
    user_input = user_msg.message.strip()

    if not user_input:
        return {"response": "I'm here to listen. Please share what's on your mind."}

    sentiment_label, _ = classify_text(user_input)
    enriched_input = f"{user_input} [{sentiment_label}]"
    chat_history.append({"from": "human", "value": enriched_input})

    prompt_text = build_prompt(chat_history)
    response = client.completions.create(
        model="tgi",
        prompt=prompt_text,
        max_tokens=80,
        temperature=0.7,
        top_p=0.9,
        stream=False
    )

    generated_text = response.choices[0].text.strip().split("\n")[0].strip()
    chat_history.append({"from": "gpt", "value": generated_text})

    return {
        "response": generated_text,
        "sentiment": sentiment_label
    }
