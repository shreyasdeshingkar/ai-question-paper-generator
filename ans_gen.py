from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv
import google.generativeai as genai
import os
import datetime
import re
import time

# Load environment variables
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv('API_KEY')
genai.configure(api_key=os.getenv("API_KEY"))

# FAISS path for book data
DB_FAISS_PATH_DATA = 'vectorstore_data/db_faiss'

'''
using random.uniform(0.5,1) ---> for maintaining randomness of the question papers.
changed model to gemini-2.0-flash
'''
import random
# Load Gemini LLM
def load_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=random.uniform(0.5,1))

# Load embeddings and FAISS DB
def load_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    book_db = FAISS.load_local(DB_FAISS_PATH_DATA, embeddings, allow_dangerous_deserialization=True)
    return book_db

# Prompt Template for answer generation
answer_prompt_template = """
You are a university-level subject expert in Computer Engineering. Your task is to provide a complete, structured, and high-quality answer key for the given question based on the context provided.
You will be given a question and relevant context from the textbook or syllabus. And Notes for answering questions.

### Question:
{question}

### Context (from textbook or syllabus):
{context}

### Answer:

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
  write correct answer only. if question asks for explanation, provide a brief justification. if question has select multiple options, list all correct options.
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

prompt = PromptTemplate(template=answer_prompt_template, input_variables=["question", "context"])

# Generate answer using LLM
def generate_answer(qp_md_path):
    book_db = load_vectorstore()
    question_paper = open(qp_md_path, 'r', encoding='utf-8').read()
    llm = load_llm()
    full_prompt = prompt.format(question=question_paper, context=book_db.as_retriever().get_relevant_documents(question_paper))
    response = llm.invoke(full_prompt)
# Main function
    # Save answers
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"answers_{timestamp}.md"
    
    output_path = None
    if "outputs" in os.listdir(os.getcwd()):
        output_path = os.path.join("outputs", filename)

    if output_path is None:
        os.makedirs("outputs", exist_ok=True)
        output_path = os.path.join("outputs", filename)

    with open(output_path, 'w', encoding='utf-8') as file:
        file.writelines(response.content)

    return output_path
