from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import requests
import time
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Therapeutic Chat API", description="API for therapeutic chat with sentiment and facial emotion analysis")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Configuration
SENTIMENT_API_URL = "https://fok73vtkey70bt77.us-east-1.aws.endpoints.huggingface.cloud"
SENTIMENT_HEADERS = {
    "Authorization": "Bearer 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",
    "Content-Type": "application/json"
}

# Facial API URL - assuming it's running on localhost:8000
FACIAL_API_URL = "http://localhost:8000/detect_emotion"

SENTIMENT_LABELS = {
    "LABEL_0": "normal",
    "LABEL_1": "depression",
    "LABEL_2": "suicidal",
    "LABEL_3": "anxiety",
    "LABEL_4": "stress",
    "LABEL_5": "bipolar",
    "LABEL_6": "personality_disorder"
}

# Emotion priority mapping (sad-like emotions take priority)
EMOTION_PRIORITY = {
    "depression": 5,
    "suicidal": 6,
    "anxiety": 4,
    "stress": 3,
    "bipolar": 4,
    "personality_disorder": 5,
    "sad": 3,
    "neutral": 1,
    "happy": 0,
    "normal": 0
}

PREDEFINED_RESPONSES = {
    "hi": "Hi, how are you doing today?",
    "hello": "Hello there, I'm here to listen. What's on your mind?",
    "hey": "Hey! What would you like to talk about today?",
    "good morning": "Good morning. How are you feeling today?",
    "good afternoon": "Good afternoon. I'm here for you — how can I support you?",
    "good evening": "Good evening. What's been on your mind lately?",
    "how are you": "I'm here and ready to support you. How are *you* doing?",
    "thanks": "You're welcome. I'm here whenever you need to talk.",
    "thank you": "You're very welcome. I'm glad to be here for you.",
    "bye": "Take care of yourself. I'm always here if you need me.",
    "goodbye": "Goodbye for now. You're not alone.",
}

# Pydantic models
class ChatMessage(BaseModel):
    session_id: str = None
    user_message: str
    capture_facial: bool = True  # Flag to enable/disable facial capture

class ChatResponse(BaseModel):
    therapist_reply: str
    sentiment: str
    facial_emotion: str = None
    final_emotion: str = None

class ChatHistory(BaseModel):
    messages: List[Dict]

# Store chat histories
chat_histories = {}

def get_sentiment_token(text: str) -> str:
    """Get sentiment analysis from text"""
    try:
        response = requests.post(SENTIMENT_API_URL, headers=SENTIMENT_HEADERS, json={"inputs": text})
        output = response.json()
        if isinstance(output, list) and len(output) > 0:
            predictions = output[0]
            predicted_label_id = predictions["label"]
            return SENTIMENT_LABELS.get(predicted_label_id, "normal")
        return "normal"
    except Exception as e:
        print(f"Sentiment API error: {e}")
        return "normal"

def get_facial_emotion(duration: int = 5) -> str:
    """Get facial emotion from facial.py API"""
    try:
        # Call the facial.py API with specified duration
        response = requests.post(f"{FACIAL_API_URL}?duration={duration}")
        if response.status_code == 200:
            result = response.json()
            if result.get("success", False) and result.get("result", {}).get("emotion"):
                return result["result"]["emotion"]
        return "neutral"  # Default if no facial emotion detected
    except Exception as e:
        print(f"Facial API error: {e}")
        return "neutral"  # Default if API fails

def combine_emotions(facial_emotion: str, sentiment: str) -> str:
    """Combine facial emotion with sentiment analysis to determine final emotional state
    Give priority to sad-like emotions based on the EMOTION_PRIORITY scale
    """
    # Handle missing values
    if not facial_emotion:
        facial_emotion = "neutral"
    if not sentiment:
        sentiment = "normal"
        
    # Get priority scores
    facial_priority = EMOTION_PRIORITY.get(facial_emotion.lower(), 0)
    sentiment_priority = EMOTION_PRIORITY.get(sentiment.lower(), 0)
    
    # Return the emotion with higher priority score
    if sentiment_priority >= facial_priority:
        return sentiment
    else:
        return facial_emotion

def build_prompt(chat_history: List[Dict]) -> str:
    prompt = "<s>"
    prompt += (
        "You are a highly trained, compassionate psychotherapist helping users navigate mental and emotional challenges. "
        "You specialize in offering empathetic, non-judgmental, trauma-informed care. "
        "Before every user message, you will see an emotion tag such as <depression>, <stress>, <anxiety>, etc. "
        "Use this tag to adjust the tone, sensitivity, and focus of your response, ensuring it is appropriate to the user's emotional state. "
        "Do not reference or repeat the tag in your response. Be gentle, validating, and supportive while guiding the user through their thoughts and feelings.\n\n"
    )
    for message in chat_history:
        if message["from"] == "human":
            prompt += f"User: {message['value']}\n"
        elif message["from"] == "gpt":
            prompt += f"Therapist: {message['value']}\n"
    prompt += "</s>\nTherapist: "
    return prompt

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    try:
        # Initialize chat history for new sessions
        if message.session_id not in chat_histories:
            chat_histories[message.session_id] = []

        # Check for predefined response
        text_lower = message.user_message.lower().strip()
        if text_lower in PREDEFINED_RESPONSES:
            predefined_reply = PREDEFINED_RESPONSES[text_lower]
            chat_histories[message.session_id].append({"from": "human", "value": f"<normal> {message.user_message}"})
            chat_histories[message.session_id].append({"from": "gpt", "value": predefined_reply})
            return ChatResponse(
                therapist_reply=predefined_reply, 
                sentiment="normal",
                facial_emotion="neutral", 
                final_emotion="normal"
            )

        # Get sentiment from text
        sentiment = get_sentiment_token(message.user_message)
        
        # Get facial emotion if enabled (call the separate facial.py API)
        facial_emotion = "neutral"
        if message.capture_facial:
            facial_emotion = get_facial_emotion(duration=5)  # 5 seconds of facial capture
        
        # Combine facial and sentiment emotions
        final_emotion = combine_emotions(facial_emotion, sentiment)
        
        # Format the final emotional state with angle brackets for the prompt
        final_emotion_token = f"<{final_emotion}>"
        
        # Combine with user message
        combined_input = f"{final_emotion_token} {message.user_message}"
        chat_histories[message.session_id].append({"from": "human", "value": combined_input})

        prompt_text = build_prompt(chat_histories[message.session_id])

        # Send generation request to HF endpoint
        generation_response = requests.post(
            url="https://h80ls72bh8kqg12h.us-east-1.aws.endpoints.huggingface.cloud/v1/completions",
            headers={
                "Authorization": "Bearer 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",
                "Content-Type": "application/json"
            },
            json={
                "model": "tgi",
                "prompt": prompt_text,
                "max_tokens": 80,
                "temperature": 0.7,
                "top_p": 0.9
            }
        )

        completion = generation_response.json()
        if "choices" in completion and len(completion["choices"]) > 0:
            generated_text = completion["choices"][0]["text"].strip().split("\n")[0].strip()
        else:
            generated_text = "I'm here to listen and support you. Could you tell me more about how you're feeling?"

        chat_histories[message.session_id].append({"from": "gpt", "value": generated_text})

        return ChatResponse(
            therapist_reply=generated_text, 
            sentiment=sentiment,
            facial_emotion=facial_emotion,
            final_emotion=final_emotion
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat-history/{session_id}", response_model=ChatHistory)
async def get_chat_history(session_id: str = "default"):
    if session_id not in chat_histories:
        return ChatHistory(messages=[])
    return ChatHistory(messages=chat_histories[message.session_id])

@app.delete("/chat-history/{session_id}")
async def clear_chat_history(session_id: str = "default"):
    if session_id in chat_histories:
        chat_histories[session_id] = []
    return {"message": "Chat history cleared"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
