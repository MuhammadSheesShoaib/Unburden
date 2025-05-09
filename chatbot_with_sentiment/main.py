import os
from openai import OpenAI
import requests

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
    prompt += ("You are a compassionate therapist providing empathetic and supportive responses. "
               "Engage with the user in a caring, thoughtful manner while offering gentle guidance.\n")
    for message in chat_history:
        if message["from"] == "human":
            prompt += f"User: {message['value']}\n"
        elif message["from"] == "gpt":
            prompt += f"Therapist: {message['value']}\n"
    prompt += eos_token + "\nTherapist: "
    return prompt

chat_history = []

greeting_responses = {
    "hi": "Hi there, I'm here to listen. How are you feeling today?",
    "hello": "Hello, I'm here for you. Would you like to share what's on your mind?"
}

print("Therapist: Hello, I'm here to listen. What's on your mind? (Type 'exit' to end the chat)")

while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        print("Therapist: Take care! I'm here whenever you need to talk.")
        break

    clean_input = user_input.lower()
    if clean_input in greeting_responses:
        reply = greeting_responses[clean_input]
        print("Therapist:", reply)
        chat_history.append({"from": "gpt", "value": reply})
        continue

    # Classify sentiment
    sentiment_label, confidence = classify_text(user_input)
    
    # Concatenate sentiment label at the end of the user input
    enriched_input = f"{user_input} [{sentiment_label}]"

    # Add enriched input to chat history
    chat_history.append({"from": "human", "value": enriched_input})

    print("DEBUG - Last message added to history:", chat_history[-1])

    # Build prompt and get response from LLM
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
    print("Therapist:", generated_text)
    chat_history.append({"from": "gpt", "value": generated_text})
