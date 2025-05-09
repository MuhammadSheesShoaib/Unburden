import requests

# Your API endpoint
API_URL = "https://fok73vtkey70bt77.us-east-1.aws.endpoints.huggingface.cloud"

# Your Hugging Face API token
headers = {
    "Authorization": "Bearer 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",  # Replace with your actual API token
    "Content-Type": "application/json"
}

# Label mapping
labels = {
    "LABEL_0": "Normal",
    "LABEL_1": "Depression",
    "LABEL_2": "Suicidal",
    "LABEL_3": "Anxiety",
    "LABEL_4": "Stress",
    "LABEL_5": "Bipolar",
    "LABEL_6": "Personality disorder"
}

def query(payload):
    """
    Sends a request to the Hugging Face API and returns the response.
    """
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def classify_text(text):
    """
    Classifies the input text using the Hugging Face API.
    Returns the predicted label and confidence score.
    """
    data = {"inputs": text}
    output = query(data)

    if isinstance(output, list) and len(output) > 0:
        predictions = output[0]  # Assuming the output is a list of predictions
        predicted_label_id = predictions["label"]  # Get the predicted label ID
        predicted_label = labels.get(predicted_label_id, "Unknown")  # Map to label name
        confidence_score = predictions["score"]  # Get the confidence score
        return predicted_label, confidence_score
    else:
        return "Error: Unexpected API response format.", None
