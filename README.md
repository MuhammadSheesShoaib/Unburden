# Unburden

**Unburden** is an **AI-driven mental wellness app** that provides interactive therapy sessions through a **real-time AI avatar**.  
It combines **verbal communication, facial emotion detection, sentiment analysis, and memory-driven conversations** to deliver an engaging and supportive user experience.  

---

## 🚀 Features  

- 🧑‍🤝‍🧑 **AI Therapy Avatar** – Real-time interactive avatar for personalized therapy sessions  
- 😊 **Facial Emotion Detection** – Powered by **MobileNetv2** to detect and respond to user emotions in real-time  
- 🙆 **Head Nod Recognition** – Position-based approach to understand non-verbal feedback  
- 💬 **Sentiment Analysis** – Fine-tuned **MentalBERT (bert-base-uncased)** on a mental health dataset for emotion-aware conversations  
- 🧠 **Memory System** – Uses **RAG + LangChain + LangGraph** for **short-term and long-term memory** in conversations  
- ☁️ **Cloud Deployment** – Models deployed on **AWS EC2**, **Hugging Face Spaces**, and **Hugging Face Inference Endpoints** with **Docker**  
- ⚡ **Backend Services** – **FastAPI APIs** for sentiment analysis, real-time facial emotion detection, and session management  
- 📱 **Mobile App** – Built with **React Native**, featuring:  
  - Firebase for authentication  
  - MongoDB for user and session data management  

---

## 🛠️ Tech Stack  

**AI & ML Models**  
- MobileNetv2 – Facial emotion detection  
- MentalBERT (fine-tuned) – Sentiment analysis  
- RAG + LangChain + LangGraph – Conversational memory  

**Backend**  
- FastAPI – API orchestration  
- Docker – Model containerization  
- AWS EC2, Hugging Face Spaces, Hugging Face Endpoints – Cloud deployment  

**Frontend**  
- React Native – Cross-platform mobile app  
- Firebase – Authentication & session handling  
- MongoDB – Database for users, conversations, and therapy history  

--
## Deployment

- Dockerized Models deployed across AWS EC2, Hugging Face Spaces, and Hugging Face Inference Endpoints
- FastAPI Backend manages API calls, model responses, and memory system
-React Native App interacts with backend services via secure APIs
