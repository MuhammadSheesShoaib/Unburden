from sentiment_final import classify_text

# Input text for classification
text = "I haven't been feeling well lately"

# Get predictions
predicted_label, confidence_score = classify_text(text)

# Display the results
if predicted_label != "Error: Unexpected API response format.":
    print(f"Predicted Label: {predicted_label}")
    print(f"Confidence Score: {confidence_score:.4f}")
else:
    print(predicted_label)
