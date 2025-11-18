#  🧠 AI-Based Question Paper Generator

An intelligent AI-powered automation system designed to generate exam question papers from syllabus PDFs and study materials using RAG (Retrieval-Augmented Generation), vector search, and LLMs (Large Language Models).

# 🚩 Problem Statement

Creating question papers manually requires deep subject understanding and careful mapping of syllabus topics to question difficulty. This process is:

Time-consuming

Repetitive

Error-prone

Not scalable

This system automates the process using AI, ensuring faster, accurate, and syllabus-aligned question generation.

# 🎯 Objective

To automate question paper generation by:

✔ Extracting syllabus content from PDFs
✔ Retrieving relevant concepts using embeddings + vector search
✔ Generating questions using LLMs
✔ Providing multiple formats (MCQ, Short Answer, Long Answer)
✔ Allowing difficulty-level customization

# 🔗 Application Link

🚀 Live Web App: (Add your link here once hosted — e.g., Render, Railway, Heroku, AWS)

Example placeholder: https://question-paper-ai-app.demo.com

# 🎥 Descriptive Video / Demo

📽 Video Walkthrough: (Add Drive / YouTube link)

Example placeholder: https://youtu.be/demo-video-link

# 📂 Dataset / Input Sources

Since this project is document-driven, the system works with:

📄 Syllabus PDFs

📚 Textbook content

📝 Notes / Reference material

Data is embedded and indexed inside a vector store.

# 🧠 Domain

📍 Education Technology | NLP | Generative AI | Automation

# 🌟 Key Features
Feature	Description
📑 PDF Text Extraction	Reads and preprocesses syllabus content
🔍 Vector Search	Uses semantic similarity instead of keyword matching
🤖 RAG Pipeline	Ensures generated questions are syllabus aligned
🎯 Difficulty Levels	Easy, Medium, Hard classification
🧩 Question Types	MCQs, Short Answer, Long Answer
🌐 Web UI	Flask-based user interface for quick interaction
🧱 Modular Architecture	Pluggable models and embeddings

# 🛠 Tech Stack
Component	Tools
Main Language	Python
Framework	LangChain, Flask
Embedding Search	FAISS
NLP Toolkit	NLTK
LLM	OpenAI / Gemini / Llama2 (configurable)
UI	HTML, CSS, Flask Templates

# 🗄 System Architecture
         ┌───────────────┐
         │  PDF Upload    │
         └───────┬───────┘
                 │
        ┌────────▼────────┐
        │ Text Extraction  │
        └────────┬────────┘
                 │
        ┌────────▼───────────┐
        │ Preprocessing + NLP │
        └────────┬───────────┘
                 │
        ┌────────▼──────────┐
        │ Embeddings + DB    │
        │      (FAISS)       │
        └────────┬──────────┘
                 │
     ┌───────────▼───────────┐
     │  LangChain RAG Model   │
     └───────────┬───────────┘
                 │
        ┌────────▼─────────┐
        │ Generated Output  │
        └───────────────────┘

# 🧪 Workflow

Upload syllabus PDF

Text extraction and sentence segmentation

Generate embeddings and match content with FAISS

Send retrieved content to LLM

Generate formatted question paper

Export / View output in UI

# 📦 Installation & Execution
#Clone repository
git clone https://github.com/yourusername/AI-QuestionPaper-Generator.git

#Navigate folder
cd AI-QuestionPaper-Generator

#Install dependencies
pip install -r requirements.txt

#Run web app
python app.py

# 🏗 Future Enhancements

📌 Export to PDF & DOCX automatically

🌍 Multi-language question support

📅 Exam templates (Unit Test / Midterm / Board Format)

🎭 User role system (Student / Teacher / Admin)

# 👨‍💻 Author

👋 Shreyas Deshingkar
📍 Satara, Maharashtra, India
📧 Email: shreyasdeshingkar@gmail.com

🔗 LinkedIn: https://www.linkedin.com/in/shreyas-deshingkar/

