import os
import uuid
import time
import tiktoken
import requests
import json
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

# Import necessary components for memory system
from langchain_core.documents import Document
from langchain_core.messages import get_buffer_string
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

# === Load environment variables ===
load_dotenv()

# === Chatbot API Setup ===
HUGGINGFACE_DEDICATED_ENDPOINT = "https://h80ls72bh8kqg12h.us-east-1.aws.endpoints.huggingface.cloud/v1/"
HUGGINGFACE_ACCESS_TOKEN = "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"

# Make sure to add your OpenAI API key in your .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "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") 

bos_token = "<s>"
eos_token = "</s>"

# Initialize OpenAI client for HuggingFace endpoint
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

# Define sentiment labels
labels = {
    "LABEL_0": "Normal",
    "LABEL_1": "Depression",
    "LABEL_2": "Suicidal",
    "LABEL_3": "Anxiety",
    "LABEL_4": "Stress",
    "LABEL_5": "Bipolar",
    "LABEL_6": "Personality disorder"
}

# === Memory System Setup ===
MEMORY_FILE = "chatbot_memories.json"

def save_memories_to_file(memory_data):
    """Save memories to a JSON file."""
    try:
        with open(MEMORY_FILE, 'w') as f:
            json.dump(memory_data, f)
    except Exception as e:
        print(f"Error saving memories to file: {e}")

def load_memories_from_file():
    """Load memories from JSON file."""
    try:
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading memories from file: {e}")
    return {}

# Initialize vector store for semantic memory
memory_data = load_memories_from_file()
recall_vector_store = InMemoryVectorStore(OpenAIEmbeddings(api_key=OPENAI_API_KEY))

# Load existing memories into vector store
if memory_data:
    for memory in memory_data.values():
        for doc in memory:
            document = Document(
                page_content=doc["content"],
                metadata=doc["metadata"]
            )
            recall_vector_store.add_documents([document])

# Define knowledge triple type for structured memories
class KnowledgeTriple(TypedDict):
    subject: str
    predicate: str
    object_: str

# === Memory Tools ===
def get_user_id(config: RunnableConfig) -> str:
    """Get user ID from config."""
    user_id = config["configurable"].get("user_id")
    if user_id is None:
        raise ValueError("User ID needs to be provided to save a memory.")
    return user_id

@tool
def save_recall_memory(memory_text: str, config: RunnableConfig) -> str:
    """Save memory to vectorstore for later semantic retrieval."""
    try:
        user_id = get_user_id(config)
        document = Document(
            page_content=memory_text,
            metadata={
                "user_id": user_id,
                "timestamp": time.time()
            },
        )
        recall_vector_store.add_documents([document])
        
        # Save to persistent storage
        if user_id not in memory_data:
            memory_data[user_id] = []
        
        memory_data[user_id].append({
            "content": memory_text,
            "metadata": document.metadata
        })
        
        save_memories_to_file(memory_data)
        return "Memory saved successfully"
    except Exception as e:
        print(f"Error saving memory: {e}")
        return "Error saving memory"

@tool
def search_recall_memories(query: str, config: RunnableConfig) -> List[str]:
    """Search for relevant memories."""
    try:
        user_id = get_user_id(config)
        
        # Get memories from persistent storage
        memories = memory_data.get(user_id, [])
        
        if not memories:
            return []
            
        # If query is empty, return recent memories
        if not query.strip():
            # Sort by timestamp and return recent ones
            memories.sort(key=lambda x: x["metadata"].get("timestamp", 0), reverse=True)
            return [mem["content"] for mem in memories[:5]]
            
        # For actual search, use vector store
        def _filter_function(doc: Document) -> bool:
            return doc.metadata.get("user_id") == user_id

        documents = recall_vector_store.similarity_search(
            query, k=5, filter=_filter_function
        )
        
        return [doc.page_content for doc in documents]
    except Exception as e:
        print(f"Error searching memories: {e}")
        return []

# === Memory State and Graph Setup ===
class State(MessagesState):
    # Add memories that will be retrieved based on the conversation context
    recall_memories: List[str]

# Define tokenizer for context window management
try:
    tokenizer = tiktoken.encoding_for_model("gpt-4o")
except:
    # Fallback tokenizer if gpt-4o isn't available
    tokenizer = tiktoken.get_encoding("cl100k_base")

def build_prompt(chat_history, recall_memories=None):
    """Build the prompt with chat history and memory"""
    prompt = bos_token
    prompt += (
        "You are a highly trained, compassionate psychotherapist helping users navigate mental and emotional challenges. "
        "You specialize in offering empathetic, non-judgmental, trauma-informed care. "
        "Before every user message, you will see an emotion tag such as <depression>, <stress>, <anxiety>, etc. "
        "Use this tag to adjust the tone, sensitivity, and focus of your response, ensuring it is appropriate to the user's emotional state.\n\n"
        "IMPORTANT INSTRUCTION ABOUT MEMORIES:\n"
        "1. When you see 'Recall memories' section, these are real previous conversations with this user.\n"
        "2. If the user's question or concern relates to any information in the recall memories, you MUST use that information to provide a contextual response.\n"
        "3. Always acknowledge and reference relevant past experiences the user has shared when responding.\n"
        "4. Show continuity in the conversation by connecting current concerns with previously shared information.\n"
        "5. If the user asks about something they mentioned before, refer to the recall memories to provide an informed response.\n"
    )
    
    # Add memories if available
    if recall_memories and len(recall_memories) > 0:
        prompt += "\n\nRecall memories (relevant information about this user from previous sessions):\n"
        for memory in recall_memories:
            prompt += f"- {memory}\n"
        
        # Debug print for memories
        print("\n=== Debug: Retrieved Memories Being Added to Prompt ===")
        print("Memories found:")
        for memory in recall_memories:
            print(f"- {memory}")
        print("===============================================\n")
    
    prompt += "\nGUIDELINES FOR USING MEMORIES:\n"
    prompt += "- If the user asks about something mentioned in recall memories, explicitly use that information in your response.\n"
    prompt += "- When responding, smoothly incorporate relevant details from recall memories to show you remember their history.\n"
    prompt += "- If user's current concern relates to past experiences in recall memories, connect these in your response.\n"
    prompt += "- While using recall memories, maintain a compassionate and supportive tone.\n\n"
    
    prompt += "Current conversation:\n"
    for message in chat_history:
        if message["from"] == "human":
            prompt += f"User: {message['value']}\n"
        elif message["from"] == "gpt":
            prompt += f"Therapist: {message['value']}\n"
    
    prompt += eos_token + "\nTherapist: "
    
    # Debug print for final prompt
    print("\n=== Debug: Final Prompt Being Sent to LLM ===")
    print(prompt)
    print("===============================================\n")
    
    return prompt

def load_memories(state: State, config: RunnableConfig) -> State:
    """Load memories for the current conversation."""
    try:
        # Get the conversation string from messages
        messages = state.get("messages", [])
        if not messages:
            return {"recall_memories": []}
            
        # Get the last message for context
        last_msg = messages[-1]
        if hasattr(last_msg, 'content'):
            convo_str = last_msg.content
        else:
            convo_str = last_msg.get("content", "")
            
        # Get user_id from config
        user_id = get_user_id(config)
        
        # Debug print before searching memories
        print("\n=== Debug: Searching Memories ===")
        print(f"User ID: {user_id}")
        print(f"Search Context: {convo_str}")
            
        # Search for relevant memories
        recall_memories = search_recall_memories.invoke(convo_str, config)
        
        # Debug print after searching memories
        print("Found memories:")
        for memory in recall_memories:
            print(f"- {memory}")
        print("===============================================\n")
        
        return {
            "recall_memories": recall_memories if recall_memories else [],
        }
    except Exception as e:
        print(f"Error loading memories: {e}")
        return {"recall_memories": []}

def route_tools(state: State):
    """Determine whether to use tools or end the conversation based on the last message."""
    try:
        msg = state["messages"][-1]
        # Handle both LangChain and dictionary message formats
        if hasattr(msg, 'tool_calls'):
            return "tools" if msg.tool_calls else END
        elif isinstance(msg, dict) and msg.get('tool_calls'):
            return "tools"
        return END
    except Exception as e:
        print(f"Error in route_tools: {e}")
        return END

# === Sentiment Classification Function ===
def classify_text(text):
    """Classify text sentiment using the sentiment API."""
    response = requests.post(SENTIMENT_API_URL, headers=SENTIMENT_HEADERS, json={"inputs": text})
    output = response.json()
    if isinstance(output, list) and len(output) > 0:
        predictions = output[0]
        label_id = predictions["label"]
        label = labels.get(label_id, "Unknown")
        score = predictions["score"]
        return label, score
    return "Unknown", None

# === Memory-Enhanced Agent Function ===
def process_with_memory(user_input, user_id, thread_id=None):
    """Process user input with memory capabilities."""
    try:
        # Generate a thread ID if not provided
        if thread_id is None:
            thread_id = str(uuid.uuid4())
        
        # Classify sentiment
        sentiment_label, confidence = classify_text(user_input)
        
        # Create memory configuration
        config = {"configurable": {"user_id": user_id, "thread_id": thread_id}}
        
        # Initialize memory graph if not already created
        global memory_graph
        if memory_graph is None:
            # Define the graph and add nodes
            builder = StateGraph(State)
            
            # Add nodes to the graph
            builder.add_node("load_memories", load_memories)
            builder.add_node("agent", agent)
            tools = [save_recall_memory, search_recall_memories]
            builder.add_node("tools", ToolNode(tools))
            
            # Add edges to the graph
            builder.add_edge(START, "load_memories")
            builder.add_edge("load_memories", "agent")
            builder.add_conditional_edges("agent", route_tools, ["tools", END])
            builder.add_edge("tools", "agent")
            
            # Compile the graph
            memory_saver = MemorySaver()
            memory_graph = builder.compile(checkpointer=memory_saver)
        
        # Process the input through the memory graph
        enriched_input = f"{user_input} [{sentiment_label}]"
        result = memory_graph.invoke(
            {"messages": [{"type": "human", "content": enriched_input}]},
            config=config
        )
        
        # Extract the response from the result
        if isinstance(result, dict) and "messages" in result:
            messages = result["messages"]
            if messages and len(messages) > 0:
                last_message = messages[-1]
                if isinstance(last_message, dict):
                    response_text = last_message.get("content", "")
                elif hasattr(last_message, 'content'):
                    response_text = last_message.content
                else:
                    response_text = str(last_message)
                return response_text, sentiment_label
        
        return "I'm here to listen. Could you tell me more?", sentiment_label
        
    except Exception as e:
        print(f"Error processing response: {e}")
        return "I'm sorry, I'm having trouble processing that. Could you rephrase?", sentiment_label

def display_memories(user_id: str):
    """Display all memories for a user."""
    try:
        # First try to load from persistent storage
        memories = memory_data.get(user_id, [])
        
        print("\n=== Current Memories ===")
        if memories:
            # Sort memories by timestamp if available
            memories.sort(key=lambda x: x["metadata"].get("timestamp", 0), reverse=True)
            for memory in memories:
                print(f"- {memory['content']}")
        else:
            print("No memories found for this session.")
        print("=====================\n")
    except Exception as e:
        print(f"Error displaying memories: {e}")
        print("No memories found or error accessing memories.")
        print("=====================\n")

def agent(state: State, config: Dict = None):
    """Process the current state and generate a response using prompt templates."""
    try:
        # Extract memories and current messages
        recall_str = (
            "<recall_memory>\n" + "\n".join(state["recall_memories"]) + "\n</recall_memory>"
        )
        
        # Build a string representation of the conversation for the HuggingFace model
        chat_history = []
        for msg in state["messages"]:
            # Handle LangChain message format
            if hasattr(msg, 'content'):
                content = msg.content
                type_ = "human" if hasattr(msg, 'type') and msg.type == "human" else "gpt"
            else:
                # Handle dictionary format
                content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
                type_ = "human" if isinstance(msg, dict) and msg.get("type") == "human" else "gpt"
            chat_history.append({"from": type_, "value": content})
        
        # Build prompt and get response from LLM
        prompt_text = build_prompt(chat_history, state["recall_memories"])
        response = client.completions.create(
            model="tgi",
            prompt=prompt_text,
            max_tokens=80,
            temperature=0.7,
            top_p=0.9,
            stream=False
        )
        generated_text = response.choices[0].text.strip().split("\n")[0].strip()
        
        # Save the conversation as memory
        if len(state["messages"]) > 0:
            last_msg = state["messages"][-1]
            if hasattr(last_msg, 'content'):
                user_message = last_msg.content
            elif isinstance(last_msg, dict):
                user_message = last_msg.get("content", "")
            else:
                user_message = str(last_msg)
                
            if user_message and config and "configurable" in config:  # Only save if we have a message and config
                # Save the full message as memory
                memory_text = f"User said: {user_message}"
                save_recall_memory.invoke(memory_text, config)
                
                # Extract and save specific memories if personal information is shared
                if "i am" in user_message.lower() or "i feel" in user_message.lower() or "i have" in user_message.lower():
                    memory_text = f"Personal context: {user_message}"
                    save_recall_memory.invoke(memory_text, config)
        
        return {"messages": [{"type": "ai", "content": generated_text}]}
        
    except Exception as e:
        print(f"Error in agent: {e}")
        return {"messages": [{"type": "ai", "content": "I'm here to listen. Could you tell me more?"}]}

# === Main Chat Application ===
def run_chatbot():
    """Main chat loop with memory-enhanced responses."""
    # Greeting responses for simple inputs
    greeting_responses = {
        "hi": "Hi there, I'm here to listen. How are you feeling today?",
        "hello": "Hello, I'm here for you. Would you like to share what's on your mind?"
    }
    
    # Initialize chat history and memory graph
    chat_history = []
    
    # Generate user ID for this session
    user_id = str(uuid.uuid4())
    thread_id = str(uuid.uuid4())
    
    # Initialize or reset memory graph for new session
    global memory_graph
    memory_graph = None
    
    print("Therapist: Hello, I'm here to listen. What's on your mind? (Type 'exit' to end the chat)")
    print("Special commands:")
    print("- Type 'show memories' to display current memories")
    print("- Type 'session:ID' to load a previous session")
    print("- Type 'exit' to end chat")
    print(f"Your session ID: {user_id}")
    
    while True:
        user_input = input("User: ")
        
        if user_input.lower() == "exit":
            print("Therapist: Take care! I'm here whenever you need to talk.")
            break
            
        if user_input.lower() == "show memories":
            display_memories(user_id)
            continue
            
        if user_input.lower().startswith("session:"):
            try:
                new_user_id = user_input.split(":", 1)[1].strip()
                user_id = new_user_id
                # Reset memory graph for new session
                memory_graph = None
                print(f"Therapist: Loaded session {user_id}. Previous memories are now accessible.")
                display_memories(user_id)
                # Clear chat history when switching sessions
                chat_history = []
                continue
            except Exception as e:
                print(f"Error loading session: {e}")
                continue
            
        clean_input = user_input.lower()
        
        # Simple response for greetings
        if clean_input in greeting_responses:
            time.sleep(1)  # Simulate thinking
            reply = greeting_responses[clean_input]
            print("Therapist:", reply)
            chat_history.append({"from": "human", "value": user_input})
            chat_history.append({"from": "gpt", "value": reply})
            continue
            
        try:
            # Process with memory system
            response, sentiment = process_with_memory(user_input, user_id, thread_id)
            print("Therapist:", response)
            
            # Update chat history
            chat_history.append({"from": "human", "value": f"{user_input} [{sentiment}]"})
            chat_history.append({"from": "gpt", "value": response})
            
        except Exception as e:
            print(f"Error in chat processing: {e}")
            print("Therapist: I apologize for the technical difficulty. Could you please rephrase that?")

# Run the chatbot if this script is executed directly
if __name__ == "__main__":
    run_chatbot()
