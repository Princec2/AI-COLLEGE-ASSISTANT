
# 📚 AI College Study Assistant

An AI-powered study assistant that helps students understand their study materials using uploaded PDFs.

## ✨ Features

- 📄 Upload and extract text from PDF files
- 🔍 Semantic search using Sentence Transformers and FAISS
- 🤖 Generate answers using a local Ollama LLM
- 📖 Display relevant PDF page sources
- 📝 Automatically generate multiple-choice quizzes (MCQs)
- ✅ Submit answers and calculate quiz scores
- 📚 Review correct answers and explanations
- 🌙 Dark-themed Streamlit interface

## 🛠️ Tech Stack

- Python
- Streamlit
- PyPDF2
- Sentence Transformers
- FAISS
- Ollama (Llama 3.2 1B)
- Requests

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd AI-COLLEGE-ASSISTANT
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install streamlit PyPDF2 faiss-cpu sentence-transformers requests
```

### 4. Install and run Ollama

Install Ollama, then download the model:

```bash
ollama pull llama3.2:1b
```

Ensure Ollama is running.

### 5. Run the application

```bash
streamlit run app.py
```

## 🚀 How to Use

1. Launch the Streamlit application.
2. Upload a study PDF.
3. Enter a question to get an AI-generated answer.
4. Check the relevant notes and PDF page sources.
5. Generate a quiz, select your answers, and submit.
6. Review your score and explanations.

## 👨‍💻 Project Status

Developed as an AI/ML learning project. Features may be improved over time.
