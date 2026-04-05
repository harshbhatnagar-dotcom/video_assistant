# 🎥 RAG-Based Video Timestamp Retrieval System

An AI-powered system that helps users find **exact timestamps in course videos** where a specific concept is taught.

Instead of watching long lectures, users can directly jump to the most relevant part using natural language queries.

---

## 🚀 Features

- 🎯 Find *exact timestamps* for concepts in videos  
- 🧠 RAG (Retrieval-Augmented Generation) based pipeline  
- 🎙️ Automatic transcription using Whisper (`large-v2`)  
- 🔍 Semantic search using embeddings  
- ⚡ Fast retrieval using cosine similarity  
- 📺 Works on long lecture videos  

---

## 🏗️ How It Works

Video (MP4) ↓ 
FFmpeg → Convert to MP3 ↓ 
Whisper (large-v2) → Transcription ↓ 
Chunking (timestamp + text) ↓ 
Embeddings (qwen3-embedding via Ollama) ↓ 
User Query ↓ 
Similarity Search (Cosine Similarity) ↓ 
Top Relevant Chunks ↓ 
LLM Response with Timestamps 🎯

