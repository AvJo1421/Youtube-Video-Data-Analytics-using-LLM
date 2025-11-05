# Youtube-Video-Data-Analytics-using-LLM
A Streamlit-powered dashboard that fetches and analyzes any YouTube channel’s data such as views, likes, comments, and upload trends using the YouTube Data API. It integrates LangChain, Chroma, and local LLMs through Ollama including Mistral, Llama3, and Gemma2 to enable AI-powered insights and natural language queries about your channel’s content.

# 🎬 YouTube Channel Analytics and Local LLM Chat App  

A Streamlit-powered application that helps you explore, analyze, and understand YouTube channel data using AI. The app combines data visualization, natural language querying, and local large language models for intelligent insights about videos, engagement, and content trends.  

## ✨ Features  

**Data Fetching and Visualization**  
Connects to the YouTube Data API to fetch video details such as title, description, views, likes, and comments.  
Displays top-performing videos and visualizes engagement trends over time.  

**AI-Powered Insights**  
Uses LangChain and a local vector database (Chroma) to store and retrieve video data.  
Integrates with local LLMs using Ollama (Mistral, Llama3, Gemma2) for natural language analysis.  
Ask questions like:  
“Which videos have the best audience retention?”  
“What kind of content gets the most engagement?”  

**Offline AI Capability**  
Works completely offline once models are downloaded through Ollama.  
No need for OpenAI API keys or external calls.  

## 🧠 Tech Stack  

**Frontend:** Streamlit  
**Backend:** Python  
**AI Frameworks:** LangChain, Ollama  
**Database:** ChromaDB (Vector Store)  
**Visualization:** Plotly, Altair  

## ⚙️ Installation  

### 1. Clone the repository  
```bash
git clone https://github.com/<your-username>/youtube-analytics-llm-app.git
cd youtube-analytics-llm-app
