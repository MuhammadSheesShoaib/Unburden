from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import torch

tokenizer = AutoTokenizer.from_pretrained("Shees7/output")
model = AutoModelForSequenceClassification.from_pretrained("Shees7/output")

app = FastAPI()

class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict_sentiment(input_data: InputText):
    try:
        labels = {
            0: "Normal",
            1: "Depression",
            2: "Suicidal",
            3: "Anxiety",
            4: "Stress",
            5: "Bipolar",
            6: "Personality disorder"
        }

        inputs = tokenizer(input_data.text, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits

        predicted_label = torch.argmax(logits, dim=1).item()
        return labels[predicted_label]
    except Exception as e:
        return None
