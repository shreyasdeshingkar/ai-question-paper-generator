import os
import random
import datetime

from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash

from dotenv import load_dotenv
import google.generativeai as genai

# LangChain / Google GenAI stuff
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

# >>> NEW IMPORTS FOR PDF <<<
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from textwrap import wrap
# >>> END NEW IMPORTS <<<

# ====== ENV & FLASK SETUP ======
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("API_KEY")
genai.configure(api_key=os.getenv("API_KEY"))

app = Flask(__name__)
app.secret_key = "super-secret-key"  # change if you want

# Paths
UPLOAD_FOLDER = "uploads"
BOOK_DB_PATH = "vectorstore_data/db_faiss"
SYLL_DB_PATH = "vectorstore_Syllabus/db_faiss"
QP_FOLDER = "generated_question_papers"
ANS_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(QP_FOLDER, exist_ok=True)
os.makedirs(ANS_FOLDER, exist_ok=True)
os.makedirs("vectorstore_data", exist_ok=True)
os.makedirs("vectorstore_Syllabus", exist_ok=True)


# >>> NEW: HELPER TO SAVE TEXT AS PDF <<<
def save_text_as_pdf(text: str, pdf_path: str, font_name: str = "Helvetica", font_size: int = 11):
    """
    Convert plain text/markdown-like content into a simple PDF.
    Handles basic line wrapping and multi-page flow.
    """
    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4
    x_margin, y_margin = 50, 50
    line_height = font_size + 3

    c.setFont(font_name, font_size)
    y = height - y_margin

    for line in text.splitlines():
        wrapped_lines = wrap(line, 100) or [""]  # preserve empty lines
        for wline in wrapped_lines:
            if y < y_margin:
                c.showPage()
                c.setFont(font_name, font_size)
                y = height - y_margin
            c.drawString(x_margin, y, wline)
            y -= line_height

    c.save()
# >>> END NEW PDF HELPER <<<


# ====== COMMON LLM / EMBEDDINGS HELPERS ======
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=random.uniform(0.5, 1)
    )

def load_embeddings_model():
    return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")


# ====== STEP 1: BUILD FAISS FROM UPLOADED PDFS ======
def build_vectorstores(syllabus_pdf_path: str, book_pdf_path: str):
    """
    - Load syllabus PDF -> store in SYLL_DB_PATH
    - Load book/notes PDF -> store in BOOK_DB_PATH
    """
    embeddings = load_embeddings_model()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    # ---- Book / Notes ----
    book_loader = PyPDFLoader(book_pdf_path)
    book_docs = book_loader.load()
    book_chunks = text_splitter.split_documents(book_docs)
    book_db = FAISS.from_documents(book_chunks, embeddings)
    book_db.save_local(BOOK_DB_PATH)

    # ---- Syllabus ----
    syll_loader = PyPDFLoader(syllabus_pdf_path)
    syll_docs = syll_loader.load()
    syll_chunks = text_splitter.split_documents(syll_docs)
    syll_db = FAISS.from_documents(syll_chunks, embeddings)
    syll_db.save_local(SYLL_DB_PATH)


# ====== QUESTION PAPER PROMPT & GEN (based on your qp_gen logic) ======

question_paper_template = """
You are an experienced professor tasked with creating a question paper for Software Engineering. Follow these instructions precisely:

1. Use the provided context and syllabus to generate questions.

2. Each question MUST be based on specific units as follows:
   - Q.1 (MCQs): From UNIT 1 to 5 
   - Q.2: From UNIT 1 only
   - Q.3: From UNIT 2 only
   - Q.4: From UNIT 3 only
   - Q.5: From UNIT 4 only
   - Q.6: From UNIT 5 only

3. Question Format Requirements:
   - Q.1: Generate 12 multiple choice questions
     * Each MCQ must have 4 options (a, b, c, d)
     * Each MCQ should end with (1)
     * Each MCQ must be seperated by 1 new line
   
   - Q.2 to Q.6: 
     * For Q.2 and Q.3: Two sub-questions (A and B)
     * Each sub-question should end with (6)
     * For Q.4, Q.5, and Q.6: Three sub-questions (A, B, and C)
     * Each sub-question should end with (6)
     * Students attempt any two from A, B, C
     * Students attempt any two from A, B


5. Question Guidelines:
   - Questions should test different cognitive levels (understanding, application, analysis)
   - Sub-questions should be related but test different aspects of the same unit
   - Questions should be clear, unambiguous, and appropriate for a 3-hour examination
   - Include a mix of theoretical and practical questions where applicable

6. General Instructions:
   - Format the entire question paper using Markdown syntax.
   - Ensure each MCQ option and sub-question is on a new line.
   - Add a blank line (normal new line) between sub-questions for proper spacing.
   - Use proper indentation for MCQ options (4 spaces for each option).
   - Add horizontal lines (---) to separate sections like Instructions, Questions, and Answers for clarity.

Use this structure for the question paper header:

<div align="center">

DR. BABASAHEB AMBEDKAR TECHNOLOGICAL UNIVERSITY, LONERE  
Regular/Supplementary Winter Examination – 2024  

Course: Computer Engineering  
Subject Code & Name: BTCOC501: Software Engineering  
Branch: Computer Engineering  
Semester: V  

Time: 3 Hours                                                                     Max. Marks: 60  

</div>

---

### Instructions:  
1. All questions are compulsory.  
2. Figures to the right indicate full marks.  
3. Assume suitable data if necessary.  

---

Context for question generation (from book/notes):
{context}

Syllabus structure:
{qp_structure}

Generate a complete question paper following all the above requirements strictly, ensuring each question/sub-question ends with marks in parentheses.
"""

qp_prompt = PromptTemplate(
    template=question_paper_template,
    input_variables=["context", "qp_structure"]
)

def load_book_and_syllabus():
    embeddings = load_embeddings_model()
    book_db = FAISS.load_local(BOOK_DB_PATH, embeddings, allow_dangerous_deserialization=True)
    syll_db = FAISS.load_local(SYLL_DB_PATH, embeddings, allow_dangerous_deserialization=True)

    # Get one representative chunk each (you can tune this)
    book_data = book_db.similarity_search("extract relevant book data")[0].page_content
    qp_structure = syll_db.similarity_search("extract relevant question paper structure")[0].page_content

    return book_data, qp_structure


def generate_question_paper_md() -> str:
    book_data, qp_structure = load_book_and_syllabus()
    llm = load_llm()
    full_prompt = qp_prompt.format(context=book_data, qp_structure=qp_structure)
    resp = llm.invoke(full_prompt)
    return resp.content if hasattr(resp, "content") else str(resp)


# >>> CHANGED: gen_qp_file now also creates PDF and returns both paths <<<
def gen_qp_file() -> tuple[str, str]:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    md_filename = f"question_paper_{timestamp}.md"
    pdf_filename = f"question_paper_{timestamp}.pdf"

    md_path = os.path.join(QP_FOLDER, md_filename)
    pdf_path = os.path.join(QP_FOLDER, pdf_filename)

    qp_md = generate_question_paper_md()
    if not qp_md.strip():
        raise ValueError("Generated question paper is empty.")

    # Save markdown
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(qp_md)

    # Save PDF version
    save_text_as_pdf(qp_md, pdf_path)

    return md_path, pdf_path
# >>> END CHANGE <<<


# ====== ANSWER GENERATION (your ans_gen logic simplified) ======

answer_prompt_template = """
You are a university-level subject expert in Computer Engineering. Your task is to provide a complete, structured, and high-quality answer key for the given question paper based on the context provided.

### Question Paper:
{question}

### Context (from textbook or syllabus):
{context}

### Answer Key:
"""

ans_prompt = PromptTemplate(
    template=answer_prompt_template,
    input_variables=["question", "context"]
)

def load_book_vectorstore():
    embeddings = load_embeddings_model()
    return FAISS.load_local(BOOK_DB_PATH, embeddings, allow_dangerous_deserialization=True)


def find_latest_qp() -> str:
    md_files = [
        os.path.join(QP_FOLDER, f)
        for f in os.listdir(QP_FOLDER)
        if f.lower().endswith(".md")
    ]
    if not md_files:
        raise FileNotFoundError("No question paper found. Please generate one first.")
    return max(md_files, key=os.path.getmtime)


def generate_answers_md(qp_md_path: str | None = None) -> str:
    if qp_md_path is None or not os.path.isfile(qp_md_path):
        qp_md_path = find_latest_qp()

    with open(qp_md_path, "r", encoding="utf-8") as f:
        question_paper = f.read()

    book_db = load_book_vectorstore()
    retriever = book_db.as_retriever()
    docs = retriever.get_relevant_documents(question_paper)
    context_text = "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))

    llm = load_llm()
    full_prompt = ans_prompt.format(question=question_paper, context=context_text)
    resp = llm.invoke(full_prompt)
    return resp.content if hasattr(resp, "content") else str(resp)


# >>> CHANGED: gen_answers_file now also creates PDF and returns both paths <<<
def gen_answers_file(qp_md_path: str | None = None) -> tuple[str, str]:
    ans_md = generate_answers_md(qp_md_path)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    md_filename = f"answers_{timestamp}.md"
    pdf_filename = f"answers_{timestamp}.pdf"

    md_path = os.path.join(ANS_FOLDER, md_filename)
    pdf_path = os.path.join(ANS_FOLDER, pdf_filename)

    # Save markdown
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(ans_md)

    # Save PDF
    save_text_as_pdf(ans_md, pdf_path)

    return md_path, pdf_path
# >>> END CHANGE <<<


# ====== FLASK ROUTES ======

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        syllabus_file = request.files.get("syllabus_pdf")
        book_file = request.files.get("book_pdf")

        if not syllabus_file or not book_file:
            flash("Please upload both syllabus and book/notes PDFs.", "error")
            return redirect(url_for("index"))

        # Save uploaded files
        syllabus_path = os.path.join(UPLOAD_FOLDER, "syllabus.pdf")
        book_path = os.path.join(UPLOAD_FOLDER, "book.pdf")
        syllabus_file.save(syllabus_path)
        book_file.save(book_path)

        try:
            # Build FAISS stores
            build_vectorstores(syllabus_path, book_path)
            flash("Vector stores built successfully from uploaded PDFs.", "success")
        except Exception as e:
            flash(f"Error while building vector stores: {e}", "error")
            return redirect(url_for("index"))

        # Generate QP and answers in one go
        try:
            # >>> CHANGED: unpack md + pdf paths <<<
            qp_md_path, qp_pdf_path = gen_qp_file()
            ans_md_path, ans_pdf_path = gen_answers_file(qp_md_path)
            # >>> END CHANGE <<<

            flash("Question paper and answer key generated successfully.", "success")
            return render_template(
                "index.html",
                qp_file=os.path.basename(qp_pdf_path),   # send PDF to template
                ans_file=os.path.basename(ans_pdf_path), # send PDF to template
            )
        except Exception as e:
            flash(f"Error during generation: {e}", "error")
            return redirect(url_for("index"))

    return render_template("index.html")


@app.route("/download/qp/<filename>")
def download_qp(filename):
    return send_from_directory(QP_FOLDER, filename, as_attachment=True)


@app.route("/download/ans/<filename>")
def download_ans(filename):
    return send_from_directory(ANS_FOLDER, filename, as_attachment=True)



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render sets PORT
    app.run(host="0.0.0.0", port=port, debug=False)
