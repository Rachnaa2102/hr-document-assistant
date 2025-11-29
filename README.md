HR Document Assistant — AI Knowledge Base Agent

An AI-powered HR knowledge assistant that helps users query company policies, onboarding guides, FAQs, and internal documents.
This agent uses local embeddings (FREE) + Groq Llama 3.1 (FREE) to generate accurate, context-based answers from uploaded or stored .txt files.

Live Demo:
👉 https://hr-document-assistant-dkb4hdnaha25h9acqpmfvs.streamlit.app

GitHub Repo:
👉 https://github.com/Rachnaa2102/hr-document-assistant

Features:
✔ Upload .txt documents (HR policies, onboarding docs, FAQs, etc.)
✔ Converts documents into vector embeddings using HuggingFace MiniLM
✔ Fast and accurate retrieval using FAISS vector store
✔ Uses Groq Llama 3.1-8B for final answer generation (0 cost!)
✔ Can answer any HR-related question using your custom knowledge base
✔ Works both with uploaded docs or preloaded docs
✔ Clean and simple UI built using Streamlit
✔ 100% FREE — No OpenAI API needed

Architecture Diagram:
👉 https://github.com/Rachnaa2102/hr-document-assistant/blob/main/Architecture%20Diagram.png

Tech Stack
**Frontend**
Streamlit

**Backend**
Python
LangChain (community)
FAISS (vector search)
Sentence-Transformers (MiniLM embeddings)

Repository Structure
hr-document-assistant/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # All Python dependencies
├── Architecture Diagram.png
└── docs/                   # Sample HR documents
      ├── company_overview.txt
      ├── hr_policy.txt
      ├── onboarding_guide.txt
      └── support_faq.txt

Setup Instructions (Run Locally)
1. Clone the repository
   git clone https://github.com/Rachnaa2102/hr-document-assistant
   cd hr-document-assistant
2. Create virtual environment
   python -m venv venv
   source venv/bin/activate      # Mac/Linux
   venv\Scripts\activate         # Windows
3. Install dependencies
   pip install -r requirements.txt
4. Set your Groq API Key
Create a .env file:
   GROQ_API_KEY=your_key_here
   (or paste in Streamlit UI)
5. Run the application
   streamlit run app.py
Application opens at:
👉 http://localhost:8501

Usage Instructions
Upload your HR .txt documents (optional)
Enter your Groq API key
Ask any HR-related question
The assistant retrieves context + generates accurate answers

Limitations
⚠ Only .txt files supported currently
⚠ Model answers only from available documents
⚠ No PDF or DOCX support yet (can be added later)
⚠ Requires internet for Groq API

Future Improvements
🔹 Add PDF & DOCX ingestion
🔹 Add chat history
🔹 Add semantic filtering & multi-doc ranking
🔹 Add voice input/output
🔹 Add admin dashboard
🔹 Save embeddings permanently

Created For
Rooman Technologies
AI Agent Development Challenge (2025)
Submitted by Rachna A

AI Model

Groq Llama 3.1-8B-Instant (FREE, ultra fast)
