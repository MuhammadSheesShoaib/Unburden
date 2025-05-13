import os
import re
import sqlite3
import requests
import time
import json
import zipfile
from datetime import datetime
from openai import OpenAI
from collections import Counter

# Optional NLTK import - we'll use simple regex if NLTK is not available
USE_NLTK = False
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    # Test if NLTK data is available
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    USE_NLTK = True
except (ImportError, LookupError, zipfile.BadZipFile):
    # If NLTK is not available or data files are missing, we'll use regex instead
    print("NLTK data not available. Using simple regex for text processing instead.")

# Hugging Face Sentiment API Details
SENTIMENT_API_URL = "https://fok73vtkey70bt77.us-east-1.aws.endpoints.huggingface.cloud"
SENTIMENT_HEADERS = {
    "Authorization": "Bearer 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",
    "Content-Type": "application/json"
}

SENTIMENT_LABELS = {
    "LABEL_0": "<normal>",
    "LABEL_1": "<depression>",
    "LABEL_2": "<suicidal>",
    "LABEL_3": "<anxiety>",
    "LABEL_4": "<stress>",
    "LABEL_5": "<bipolar>",
    "LABEL_6": "<personality_disorder>"
}

# Predefined responses for common greetings or statements
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

HUGGINGFACE_DEDICATED_ENDPOINT = "https://h80ls72bh8kqg12h.us-east-1.aws.endpoints.huggingface.cloud/v1/"
HUGGINGFACE_ACCESS_TOKEN = "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"

bos_token = "<s>"
eos_token = "</s>"

# Memory database setup
class MemoryManager:
    def __init__(self, db_path="memory.db"):
        """Initialize the memory manager with a SQLite database."""
        self.db_path = db_path
        self.setup_database()
        
    def setup_database(self):
        """Set up the memory database schema if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create memory table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            emotion TEXT,
            topic TEXT,
            summary TEXT,
            keywords TEXT,
            importance INTEGER DEFAULT 1
        )
        ''')

        # Create keywords table for faster retrieval
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS keywords (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id INTEGER,
            keyword TEXT,
            FOREIGN KEY (memory_id) REFERENCES memories(id)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_memory(self, user_id, emotion, message, bot_response):
        """Save a memory after a conversation exchange."""
        # Extract topics and keywords from the message
        topic = self._extract_topic(message)
        keywords = self._extract_keywords(message)
        
        # Create a summary from the conversation
        summary = f"User mentioned {topic} saying '{self._truncate(message, 50)}'. Bot responded with '{self._truncate(bot_response, 50)}'"
        
        # Calculate importance (1-5) based on message length, emotion type, etc.
        importance = self._calculate_importance(emotion, message)
        
        # Save to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO memories (user_id, emotion, topic, summary, keywords, importance)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (user_id, emotion, topic, summary, json.dumps(keywords), importance))
        
        memory_id = cursor.lastrowid
        
        # Save individual keywords for faster retrieval
        for keyword in keywords:
            cursor.execute('INSERT INTO keywords (memory_id, keyword) VALUES (?, ?)', 
                          (memory_id, keyword))
        
        conn.commit()
        conn.close()
        
        return memory_id
    
    def retrieve_relevant_memories(self, user_id, current_message, limit=3):
        """Retrieve memories relevant to the current message."""
        # Extract keywords from current message
        current_keywords = self._extract_keywords(current_message)
        if not current_keywords:
            return []
            
        # Identify emotion from current message
        emotion = self._identify_emotion(current_message)
        
        # Query database for memories with matching keywords or emotion
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        memories = []
        
        # First try: exact keyword matches
        if current_keywords:
            placeholders = ','.join(['?' for _ in current_keywords])
            query = f'''
            SELECT m.*, COUNT(k.keyword) as match_count
            FROM memories m
            JOIN keywords k ON m.id = k.memory_id
            WHERE m.user_id = ? 
            AND k.keyword IN ({placeholders})
            GROUP BY m.id
            ORDER BY match_count DESC, m.importance DESC, m.timestamp DESC
            LIMIT ?
            '''
            
            cursor.execute(query, (user_id, *current_keywords, limit))
            memories = [dict(row) for row in cursor.fetchall()]
        
        # Second try: topic matches if no keyword matches
        if not memories:
            # Extract topic from current message
            topic = self._extract_topic(current_message)
            
            cursor.execute('''
            SELECT * FROM memories 
            WHERE user_id = ? AND topic = ?
            ORDER BY importance DESC, timestamp DESC
            LIMIT ?
            ''', (user_id, topic, limit))
            memories = [dict(row) for row in cursor.fetchall()]
        
        # Third try: emotion match if still no matches
        if not memories and emotion:
            cursor.execute('''
            SELECT * FROM memories 
            WHERE user_id = ? AND emotion = ?
            ORDER BY importance DESC, timestamp DESC
            LIMIT ?
            ''', (user_id, emotion, limit))
            memories = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        
        # Format memories for prompt injection
        formatted_memories = []
        for memory in memories:
            formatted_date = datetime.fromisoformat(memory['timestamp']).strftime("%Y-%m-%d")
            formatted_memories.append({
                "date": formatted_date,
                "topic": memory['topic'],
                "summary": memory['summary'],
                "emotion": memory['emotion']
            })
            
        return formatted_memories
    
    def _extract_topic(self, message):
        """Extract the main topic from a message."""
        # Simple implementation - find the most common noun or use the first few words
        # This could be enhanced with NLP techniques
        words = message.lower().split()
        if not words:
            return "general"
            
        # Common mental health topics that might appear in this chatbot
        topics = ["anxiety", "depression", "stress", "work", "family", "relationship", 
                 "sleep", "mood", "therapy", "medication", "coping", "health"]
        
        for topic in topics:
            if topic in message.lower():
                return topic
                
        # If no specific topic found, return first few words
        return " ".join(words[:2])
    
    def _extract_keywords(self, message, max_keywords=5):
        """Extract keywords from a message."""
        if USE_NLTK:
            # Use NLTK if available
            stop_words = set(stopwords.words('english'))
            tokens = word_tokenize(message.lower())
            
            # Filter out stopwords and punctuation
            keywords = [word for word in tokens if word.isalnum() and word not in stop_words]
        else:
            # Fallback to simple regex approach
            # Remove punctuation and convert to lowercase
            clean_message = re.sub(r'[^\w\s]', '', message.lower())
            
            # Define simple stopwords list
            simple_stopwords = {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
                "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 
                'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 
                'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 
                'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 
                'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 
                'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 
                'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 
                'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 
                'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 
                'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 
                'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 
                'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
                'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 
                'now', 'd', 'll', 'm', 'o', 're', 've', 'y'
            }
            
            # Split into words and filter out stopwords
            words = clean_message.split()
            keywords = [word for word in words if word not in simple_stopwords and len(word) > 2]
        
        # Count frequency of each word
        word_counts = Counter(keywords)
        
        # Return the most common words
        return [word for word, count in word_counts.most_common(max_keywords)]
    
    def _identify_emotion(self, message):
        """Simple rule-based emotion identification."""
        emotion_keywords = {
            "<depression>": ["sad", "depressed", "hopeless", "empty", "worthless"],
            "<anxiety>": ["anxious", "worried", "nervous", "panic", "fear"],
            "<stress>": ["stressed", "overwhelmed", "pressure", "tension"],
            "<normal>": ["okay", "fine", "normal", "good", "well"],
        }
        
        message_lower = message.lower()
        for emotion, keywords in emotion_keywords.items():
            for keyword in keywords:
                if keyword in message_lower:
                    return emotion
                    
        return None
    
    def _calculate_importance(self, emotion, message):
        """Calculate importance of a memory (1-5)."""
        importance = 1
        
        # Longer messages might be more important
        if len(message) > 100:
            importance += 1
            
        # Certain emotions might indicate more important memories
        if emotion in ["<depression>", "<suicidal>", "<anxiety>"]:
            importance += 1
            
        # Messages with specific keywords might be more important
        important_keywords = ["always", "never", "hate", "love", "childhood", 
                             "trauma", "abuse", "family", "help", "scared", "terrified"]
        for keyword in important_keywords:
            if keyword in message.lower():
                importance += 1
                break
                
        return min(importance, 5)  # Cap at 5
    
    def _truncate(self, text, max_length=50):
        """Truncate a string to a maximum length."""
        if len(text) <= max_length:
            return text
        return text[:max_length] + "..."


# Match predefined user input (punctuation-insensitive)
def match_predefined_response(user_input):
    cleaned = re.sub(r'[^\w\s]', '', user_input.lower().strip())
    return PREDEFINED_RESPONSES.get(cleaned, None)

def query_sentiment(payload):
    response = requests.post(SENTIMENT_API_URL, headers=SENTIMENT_HEADERS, json=payload)
    return response.json()

def get_sentiment_token(text):
    data = {"inputs": text}
    output = query_sentiment(data)
    if isinstance(output, list) and len(output) > 0:
        predictions = output[0]
        predicted_label_id = predictions["label"]
        return SENTIMENT_LABELS.get(predicted_label_id, "<unknown>")
    else:
        return "<error>"

def build_prompt(chat_history, user_id, current_message, memory_manager):
    """Build prompt with relevant memories injected."""
    # Retrieve relevant memories
    relevant_memories = memory_manager.retrieve_relevant_memories(
        user_id, current_message, limit=3
    )
    
    prompt = bos_token
    prompt += (
        "You are a highly trained, compassionate psychotherapist helping users navigate mental and emotional challenges. "
        "You specialize in offering empathetic, non-judgmental, trauma-informed care. "
        "Before every user message, you will see an emotion tag such as <depression>, <stress>, <anxiety>, etc. "
        "Use this tag to adjust the tone, sensitivity, and focus of your response, ensuring it is appropriate to the user's emotional state. "
        "Do not reference or repeat the tag in your response. Be gentle, validating, and supportive while guiding the user through their thoughts and feelings.\n\n"
    )
    
    # Inject relevant memories if available
    if relevant_memories:
        prompt += "Previous relevant memories from this user (use this information to provide more personalized responses):\n"
        for memory in relevant_memories:
            prompt += f"- [{memory['date']}] Topic: {memory['topic']} | Emotion: {memory['emotion']} | {memory['summary']}\n"
        prompt += "\nRemember the above information about the user's history but DO NOT explicitly mention that you remember their previous conversations unless they ask.\n\n"
    
    # Add conversation history
    for message in chat_history:
        if message["from"] == "human":
            prompt += f"User: {message['value']}\n"
        elif message["from"] == "gpt":
            prompt += f"Therapist: {message['value']}\n"
    
    prompt += eos_token + "\nTherapist: "
    return prompt

def main():
    # Initialize components
    memory_manager = MemoryManager("memory.db")
    
    # Initialize OpenAI-compatible client for HF endpoint
    client = OpenAI(
        base_url=HUGGINGFACE_DEDICATED_ENDPOINT,
        api_key=HUGGINGFACE_ACCESS_TOKEN,
    )
    
    chat_history = []
    
    # For demo purposes, use a fixed user ID
    user_id = "user123"
    
    print("Therapist: Hello, I'm here to listen. What's on your mind? (Type 'exit' to end the chat)")
    
    while True:
        user_input = input("User: ")
        if user_input.lower().strip() == "exit":
            print("Therapist: Take care! I'm here whenever you need to talk.")
            break
    
        # Check for predefined response
        predefined_reply = match_predefined_response(user_input)
        if predefined_reply:
            time.sleep(2)  # Delay for realism
            print("Therapist:", predefined_reply)
            
            # Add to chat history
            sentiment_token = "<normal>"
            combined_input = f"{sentiment_token} {user_input}"
            chat_history.append({"from": "human", "value": combined_input})
            chat_history.append({"from": "gpt", "value": predefined_reply})
            
            # Save memory from this exchange
            memory_manager.save_memory(user_id, sentiment_token, user_input, predefined_reply)
            continue
    
        # Handle sentiment + model response
        sentiment_token = get_sentiment_token(user_input)
        combined_input = f"{sentiment_token} {user_input}"
        chat_history.append({"from": "human", "value": combined_input})
    
        # Build prompt with memory injection
        prompt_text = build_prompt(chat_history, user_id, user_input, memory_manager)
    
        time.sleep(3)  # Delay before model response to feel human
    
        response = client.completions.create(
             model="tgi",
             prompt=prompt_text,
             max_tokens=80,
             temperature=0.7,
             top_p=0.9,
             stream=False
        )
    
        if hasattr(response, "choices") and len(response.choices) > 0:
            generated_text = response.choices[0].text.strip().split("\n")[0].strip()
        else:
            generated_text = "<error>"
    
        print("Therapist:", generated_text)
        chat_history.append({"from": "gpt", "value": generated_text})
        
        # Save memory from this exchange
        memory_manager.save_memory(user_id, sentiment_token, user_input, generated_text)

main()
