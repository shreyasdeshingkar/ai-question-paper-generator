# from langchain.prompts import PromptTemplate
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain_community.vectorstores import FAISS
# from langchain_core.documents import Document
# from dotenv import load_dotenv
# import google.generativeai as genai
# import os
# import datetime
# import re
# import time

# # Load environment variables
# load_dotenv()
# os.environ['GOOGLE_API_KEY'] = os.getenv('API_KEY')
# genai.configure(api_key=os.getenv("API_KEY"))

# # FAISS path for book data
# DB_FAISS_PATH_DATA = 'vectorstore_data/db_faiss'

# '''
# using random.uniform(0.5,1) ---> for maintaining randomness of the question papers.
# changed model to gemini-2.0-flash
# '''
# import random
# # Load Gemini LLM
# def load_llm():
#     return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=random.uniform(0.5,1))

# # Load embeddings and FAISS DB
# def load_vectorstore():
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
#     book_db = FAISS.load_local(DB_FAISS_PATH_DATA, embeddings, allow_dangerous_deserialization=True)
#     return book_db

# # Prompt Template for answer generation
# answer_prompt_template = """
# You are a university-level subject expert in Computer Engineering. Your task is to provide a complete, structured, and high-quality answer key for the given question based on the context provided.
# You will be given a question and relevant context from the textbook or syllabus. And Notes for answering questions.

# ### Question:
# {question}

# ### Context (from textbook or syllabus):
# {context}

# ### Answer:

# # Notes:
# - ## What Type of Answers Are Expected? To fairly award marks, professors typically look for the following in a **6-mark question**:

# ### 1. Relevance to the Question
# - The answer must **directly address** what is being asked.
# - Avoid writing generic content or off-topic theory.

# ### 2. Conceptual Clarity
# - The student should demonstrate **understanding** of the concept, not just memorization.
# - Definitions, examples, or brief explanations are preferred over vague generalizations.

# ### 3. Structure and Presentation
# - Good answers usually follow this structure:
#   - **Definition or Introduction** (1 mark)
#   - **Explanation** (2–3 marks)
#   - **Example / Diagram** (if applicable) (1–2 marks)
# - **Use of bullet points, headings, or diagrams** can enhance readability and may influence partial marks.

# ### 4. Solving MCQs
# - For **Multiple Choice Questions (MCQs)**,
#   write correct answer only. if question asks for explanation, provide a brief justification. if question has select multiple options, list all correct options.
# ---

# ## Minimum Suitable Length for a 6-Mark Answer

# While quality matters more than length, the answer should not be too short. Typically, a 6-mark answer should:

# - Be **at least 3/4th to 1 full page** in a standard answer sheet
# - Include:
#   - 1–2 paragraphs of explanation
#   - A relevant **example or diagram**
# - Around **100–150 words** minimum is expected

# **Too short (1–2 lines)** → Max 1 or 2 marks  
# **Well-structured and detailed** → Full 6 marks possible

# ---

# ## Common Professor Expectations
# - **Use of terminology:** Use terms like “elicitation,” “stakeholders,” “use case,” “state transition,” etc., accurately.
# - **Neat diagrams:** Especially in questions involving UML, behavioral models, or architecture.
# - **Avoid fluff:** Unnecessary repetition or filler content may reduce marks.

# ---

# ## Additional Evaluation Tips

# - For **"Any TWO"** type questions, if the student answers all three, only the **first two** will be evaluated unless specified.
# - If a diagram is **required** (e.g., State Diagram, Context Model) and is **not drawn**, up to 2 marks may be deducted.
# - **Examples** must be relevant; incorrect examples can reduce scores.


# """

# prompt = PromptTemplate(template=answer_prompt_template, input_variables=["question", "context"])

# # Generate answer using LLM
# def generate_answer(qp_md_path):
#     book_db = load_vectorstore()
#     question_paper = open(qp_md_path, 'r', encoding='utf-8').read()
#     llm = load_llm()
#     full_prompt = prompt.format(question=question_paper, context=book_db.as_retriever().get_relevant_documents(question_paper))
#     response = llm.invoke(full_prompt)
# # Main function
#     # Save answers
#     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = f"answers_{timestamp}.md"
    
#     output_path = None
#     if "outputs" in os.listdir(os.getcwd()):
#         output_path = os.path.join("outputs", filename)

#     if output_path is None:
#         os.makedirs("outputs", exist_ok=True)
#         output_path = os.path.join("outputs", filename)

#     with open(output_path, 'w', encoding='utf-8') as file:
#         file.writelines(response.content)

#     return output_path



from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv
import google.generativeai as genai
import os
import datetime
import random

# ===============================
# ENV & CONFIG
# ===============================
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv('API_KEY')
genai.configure(api_key=os.getenv("API_KEY"))

# FAISS path for book data
DB_FAISS_PATH_DATA = 'vectorstore_data/db_faiss'

# ===============================
# LLM LOADER
# ===============================
def load_llm():
    """
    Load Gemini LLM with slight randomness in temperature.
    """
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=random.uniform(0.5, 1)
    )

# ===============================
# VECTORSTORE LOADER
# ===============================
def load_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    book_db = FAISS.load_local(
        DB_FAISS_PATH_DATA,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return book_db

# ===============================
# PROMPT TEMPLATE FOR ANSWERS
# ===============================
answer_prompt_template = """
You are a university-level subject expert in Computer Engineering. Your task is to provide a complete, structured, and high-quality answer key for the given question paper based on the context provided.
You will be given the entire question paper and relevant context from the textbook or syllabus, and notes for answering questions.

### Question Paper:
{question}

### Context (from textbook or syllabus):
{context}

### Answer Key:

# Notes:
- ## What Type of Answers Are Expected? To fairly award marks, professors typically look for the following in a **6-mark question**:

### 1. Relevance to the Question
- The answer must **directly address** what is being asked.
- Avoid writing generic content or off-topic theory.

### 2. Conceptual Clarity
- The student should demonstrate **understanding** of the concept, not just memorization.
- Definitions, examples, or brief explanations are preferred over vague generalizations.

### 3. Structure and Presentation
- Good answers usually follow this structure:
  - **Definition or Introduction** (1 mark)
  - **Explanation** (2–3 marks)
  - **Example / Diagram** (if applicable) (1–2 marks)
- **Use of bullet points, headings, or diagrams** can enhance readability and may influence partial marks.

### 4. Solving MCQs
- For **Multiple Choice Questions (MCQs)**,
  - Write only the correct option (like: Q1: b) or "Q1: Option (b) – Design").
  - If explanation is asked, provide a brief justification.
  - For "select multiple options" type questions, list all correct options.

---

## Minimum Suitable Length for a 6-Mark Answer

While quality matters more than length, the answer should not be too short. Typically, a 6-mark answer should:

- Be **at least 3/4th to 1 full page** in a standard answer sheet
- Include:
  - 1–2 paragraphs of explanation
  - A relevant **example or diagram**
- Around **100–150 words** minimum is expected

**Too short (1–2 lines)** → Max 1 or 2 marks  
**Well-structured and detailed** → Full 6 marks possible

---

## Common Professor Expectations
- **Use of terminology:** Use terms like “elicitation,” “stakeholders,” “use case,” “state transition,” etc., accurately.
- **Neat diagrams:** Especially in questions involving UML, behavioral models, or architecture.
- **Avoid fluff:** Unnecessary repetition or filler content may reduce marks.

---

## Additional Evaluation Tips

- For **"Any TWO"** type questions, if the student answers all three, only the **first two** will be evaluated unless specified.
- If a diagram is **required** (e.g., State Diagram, Context Model) and is **not drawn**, up to 2 marks may be deducted.
- **Examples** must be relevant; incorrect examples can reduce scores.
"""

prompt = PromptTemplate(
    template=answer_prompt_template,
    input_variables=["question", "context"]
)

# ===============================
# HELPER – FIND LATEST QUESTION PAPER
# ===============================
def _find_latest_question_paper(search_base: str = "generated_question_papers") -> str:
    """
    Find the latest .md file in the given directory (used as the latest question paper).
    Raises FileNotFoundError if none is found.
    """
    if not os.path.isdir(search_base):
        raise FileNotFoundError(
            f"Question paper directory '{search_base}' does not exist. "
            "Generate a question paper before calling generate_answer()."
        )

    md_files = [
        os.path.join(search_base, f)
        for f in os.listdir(search_base)
        if f.lower().endswith(".md")
    ]

    if not md_files:
        raise FileNotFoundError(
            f"No .md question paper found in '{search_base}'. "
            "Ensure your question paper generator saves .md files there."
        )

    latest_file = max(md_files, key=os.path.getmtime)
    return latest_file

# ===============================
# MAIN: GENERATE ANSWER KEY
# ===============================
def generate_answer(qp_md_path: str | None = None) -> str:
    """
    Generate an answer key for a question paper.

    - If qp_md_path is a valid file path -> use it.
    - If qp_md_path is a directory -> search latest .md inside it.
    - If qp_md_path is None or invalid -> automatically use the latest .md
      from the 'generated_question_papers' directory.
    """
    # Resolve question paper path
    if qp_md_path is not None and os.path.isfile(qp_md_path):
        resolved_qp_path = qp_md_path
    else:
        if qp_md_path is not None and os.path.isdir(qp_md_path):
            resolved_qp_path = _find_latest_question_paper(qp_md_path)
        else:
            # fallback to default folder where qp_gen.py saves question papers
            resolved_qp_path = _find_latest_question_paper("generated_question_papers")

    # Read question paper
    with open(resolved_qp_path, 'r', encoding='utf-8') as f:
        question_paper = f.read()

    # Load vectorstore and retrieve context
    book_db = load_vectorstore()
    retriever = book_db.as_retriever()
    context_docs = retriever.get_relevant_documents(question_paper)

    # Build context text
    context_text = "\n\n".join(
        f"[{i+1}] {doc.page_content}" for i, doc in enumerate(context_docs)
    )

    # Call LLM
    llm = load_llm()
    full_prompt = prompt.format(
        question=question_paper,
        context=context_text
    )
    response = llm.invoke(full_prompt)

    # Prepare answers output path
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"answers_{timestamp}.md"
    os.makedirs("outputs", exist_ok=True)
    output_path = os.path.join("outputs", filename)

    # Save answers
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(response.content if hasattr(response, "content") else str(response))

    return output_path
