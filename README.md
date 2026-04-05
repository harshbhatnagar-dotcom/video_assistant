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

## 📌 Description

1. **Video Input**: Start with an MP4 video file.  
2. **Audio Extraction**: Use FFmpeg to convert video into MP3 format.  
3. **Transcription**: Apply Whisper (large-v2) to generate text from audio.  
4. **Chunking**: Split transcript into smaller chunks with timestamps.  
5. **Embedding Generation**: Convert chunks into vector embeddings using qwen3 via Ollama.  
6. **User Query**: Accept a question from the user.  
7. **Similarity Search**: Perform cosine similarity to find relevant chunks.  
8. **Context Retrieval**: Select top matching chunks.  
9. **LLM Response**: Generate an answer with precise timestamps.

