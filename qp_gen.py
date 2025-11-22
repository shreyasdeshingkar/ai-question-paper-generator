# from langchain.prompts import PromptTemplate
# import google.generativeai as genai
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_community.vectorstores import FAISS
# import os
# from dotenv import load_dotenv
# from langchain_core.documents import Document
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.runnables import RunnablePassthrough
# import os
# from datetime import datetime

# # Setting Google API Key
# load_dotenv()
# os.environ['GOOGLE_API_KEY'] = os.getenv('API_KEY')
# genai.configure(api_key=os.getenv("API_KEY"))

# # Path of vectore database
# DB_FAISS_PATH_DATA = 'vectorstore_data/db_faiss'
# DB_FAISS_PATH_QP = 'vectorstore_Syllabus/db_faiss'

# '''
# using random.uniform(0.5,1) ---> for maintaining randomness of the question papers.
# changed model to gemini-2.0-flash
# '''
# import random

# # Set up Google LLM
# def load_llm():
#     llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=random.uniform(0.5,1))
#     return llm

# # Create a custom prompt template
# question_paper_template = """
# You are an experienced professor tasked with creating a question paper for Software Engineering. Follow these instructions precisely:

# 1. Use the provided context and syllabus to generate questions.

# 2. Each question MUST be based on specific units as follows:
#    - Q.1 (MCQs): From UNIT 1 to 5 
#    - Q.2: From UNIT 1 only
#    - Q.3: From UNIT 2 only
#    - Q.4: From UNIT 3 only
#    - Q.5: From UNIT 4 only
#    - Q.6: From UNIT 5 only

# 3. Question Format Requirements:
#    - Q.1: Generate 12 multiple choice questions
#      * Each MCQ must have 4 options (a, b, c, d)
#      * Each MCQ should end with (1)
#      * Each MCQ must be seperated by 1 new line
   
#    - Q.2 to Q.6: 
#      * For Q.2 and Q.3: Two sub-questions (A and B)
#      * Each sub-question should end with (6)
#      * For Q.4, Q.5, and Q.6: Three sub-questions (A, B, and C)
#      * Each sub-question should end with (6)
#      * Students attempt any two from A, B, C
#      * Students attempt any two from A, B


# 4. Question Format Example:
#    Q.2 Solve the following:
#    A) Explain the concept of requirements engineering and its importance in software development. (6)
#    B) Discuss various requirements elicitation techniques with examples. (6)

#    Q.4 Solve any TWO of the following:
#    A) What is system modeling? Explain its significance in software development. (6)
#    B) Describe the different types of UML diagrams with their purposes. (6)
#    C) Explain the concept of behavioral modeling with suitable examples. (6)

# 5. Question Guidelines:
#    - Questions should test different cognitive levels (understanding, application, analysis)
#    - Sub-questions should be related but test different aspects of the same unit
#    - Questions should be clear, unambiguous, and appropriate for a 3-hour examination
#    - Include a mix of theoretical and practical questions where applicable

# 6. General Instructions:
#    - Format the entire question paper using Markdown syntax.
#    - Ensure each MCQ option and sub-question is on a new line.
#    - Add a blank line (normal new line) between sub-questions for proper spacing.
#    - Use proper indentation for MCQ options (4 spaces for each option).
#    - Add horizontal lines (---) to separate sections like Instructions, Questions, and Answers for clarity.

# Use this structure for the question paper:

# <div align="center">

# DR. BABASAHEB AMBEDKAR TECHNOLOGICAL UNIVERSITY, LONERE  
# Regular/Supplementary Winter Examination – 2024  

# Course: Computer Engineering  
# Subject Code & Name: BTCOC501: Software Engineering  
# Branch: Computer Engineering  
# Semester: V  

# Time: 3 Hours                                                                     Max. Marks: 60  

# </div>

# ---

# ### Instructions:  
# 1. All questions are compulsory.  
# 2. Figures to the right indicate full marks.  
# 3. Assume suitable data if necessary.  

# ---


# [Rest of the header and instructions as provided]

# Context for question generation:
# {context}

# Syllabus structure:
# {qp_structure}

# Generate a complete question paper following all the above requirements strictly, ensuring each question/sub-question ends with marks in parentheses.

# Example Output Format:
# <div align="center">

# DR. BABASAHEB AMBEDKAR TECHNOLOGICAL UNIVERSITY, LONERE  
# Regular/Supplementary Winter Examination – 2024  

# Course: Computer Engineering  
# Subject Code & Name: BTCOC501: Software Engineering  
# Branch: Computer Engineering  
# Semester: V  

# Time: 3 Hours                                                                     Max. Marks: 60  

# </div>

# ---

# ### Instructions:  
# 1. All questions are compulsory.  
# 2. Figures to the right indicate full marks.  
# 3. Assume suitable data if necessary.  

# ---
# <h3>Q.1 Choose the correct answer for the following Multiple Choice Questions.</h3>
# <ol>
#   <li>Which of the following is NOT a phase in the Software Development Life Cycle (SDLC)?<br>
#     a) Requirement Gathering<br>
#     b) Design<br>
#     c) Testing<br>
#     d) Hardware Procurement
#   </li><ol>
#   <li>What is the primary goal of software engineering?<br>
#     a) To write complex code<br>
#     b) To manage software complexity<br>
#     c) To create visually appealing interfaces<br>
#     d) To maximize programmer productivity
#   </li>
#   <li>Which of the following is an example of a non-functional requirement?<br>
#     a) The system shall allow users to log in with a username and password.<br>
#     b) The system shall generate a report of all sales transactions.<br>
#     c) The system shall respond to user requests within 2 seconds.<br>
#     d) The system shall calculate the total cost of items in a shopping cart.
#   </li>
#   <li>What is the purpose of requirements elicitation?<br>
#     a) To validate the software design.<br>
#     b) To gather requirements from stakeholders.<br>
#     c) To test the software functionality.<br>
#     d) To manage the software development team.
#   </li>
#   <li>Which validation technique involves presenting the requirements to users for feedback?<br>
#     a) Inspection<br>
#     b) Prototyping<br>
#     c) Formal Verification<br>
#     d) Dataflow Diagramming
#   </li>
#   <li>What is the main goal of Requirements Management?<br>
#     a) To ensure that requirements are complete and correct<br>
#     b) To control changes to requirements throughout the project lifecycle<br>
#     c) To prioritize requirements based on stakeholder needs<br>
#     d) All of the above
#   </li>
#   <li>Which of the following is NOT a type of system model?<br>
#     a) Context Model<br>
#     b) Interaction Model<br>
#     c) Structural Model<br>
#     d) Implementation Model
#   </li>
#   <li>What does a use case diagram represent?<br>
#     a) The data flow within a system<br>
#     b) The interaction between actors and the system<br>
#     c) The structure of the system's database<br>
#     d) The sequence of events in a system
#   </li>
#   <li>Which model focuses on the dynamic behavior of the system?<br>
#     a) Structural Model<br>
#     b) Behavioral Model<br>
#     c) Context Model<br>
#     d) Data Model
#   </li>
#   <li>What is the primary goal of Model-Driven Architecture (MDA)?<br>
#     a) To automate code generation from models<br>
#     b) To improve software performance<br>
#     c) To simplify software testing<br>
#     d) To enhance user interface design
#   </li>
#   <li>What is the role of patterns in software design?<br>
#     a) To provide reusable solutions to common design problems<br>
#     b) To enforce strict coding standards<br>
#     c) To optimize code for performance<br>
#     d) To create visually appealing user interfaces
#   </li>
#   <li>Which of the following is NOT a design pattern?<br>
#     a) Singleton<br>
#     b) Factory<br>
#     c) Compiler<br>
#     d) Observer
#   </li>
# </ol>

# <h3>Q.2 Solve the following:</h3>
# <ol type="A">
#   <li>Explain the importance of a well-defined Software Requirements Document (SRD) and its typical structure. (6)</li>
#   <li>Differentiate between functional and non-functional requirements with suitable examples. (6)</li>
# </ol>

# <h3>Q.3 Solve the following:</h3>
# <ol type="A">
#   <li>Describe the different activities involved in the Requirements Engineering process. (6)</li>
#   <li>Discuss various challenges in requirements elicitation and how to address them. (6)</li>
# </ol>

# <h3>Q.4 Solve any TWO of the following:</h3>
# <ol type="A">
#   <li>What are Context Models? Explain their purpose and how they are created. (6)</li>
#   <li>Describe the different types of UML diagrams used for structural modeling, including class diagrams and component diagrams. (6)</li>
#   <li>Explain the concept of Behavioral Modeling with suitable examples, including state diagrams and activity diagrams. (6)</li>
# </ol>

# <h3>Q.5 Solve any TWO of the following:</h3>
# <ol type="A">
#   <li>What are architectural design decisions and why are they important? Explain with examples. (6)</li>
#   <li>Describe the different architectural styles, such as layered architecture, client-server architecture, and microservices architecture. (6)</li>
#   <li>Explain the importance of design patterns in software architecture and provide examples of commonly used architectural patterns. (6)</li>
# </ol>

# <h3>Q.6 Solve any TWO of the following:</h3>
# <ol type="A">
#   <li>Explain the different types of design patterns, including creational, structural, and behavioral patterns. Provide examples of each. (6)</li>
#   <li>Describe the benefits of using design patterns in software development, such as reusability, maintainability, and flexibility. (6)</li>
#   <li>Discuss the challenges of implementing design patterns and how to overcome them. (6)</li>
# </ol>


# <div align="center">
# <h2>Best of Luck!</h2>
# </div>

# """

# def generate_question_paper(book_data, qp_structure):
#     prompt = PromptTemplate(template=question_paper_template, input_variables=['context', 'qp_structure'])
#     llm = load_llm()
    
#     # Create the chain using the newer approach
#     document_chain = create_stuff_documents_chain(
#         llm=llm,
#         prompt=prompt,
#         document_variable_name="context"
#     )
    
#     # Create a Document object from the context
#     context = book_data + "\n\n" + qp_structure
#     doc = Document(page_content=context)
    
#     # Create a runnable pipeline
#     chain = (
#         RunnablePassthrough.assign(context=lambda x: x["input_documents"])
#         | document_chain
#     )
    
#     # Invoke the chain
#     response = chain.invoke({
#         "input_documents": [doc],
#         "qp_structure": qp_structure
#     })
    
#     return response


# # Load the embeddings and data
# def load_embeddings():
#     embeddings = GoogleGenerativeAIEmbeddings(model = "models/text-embedding-004")
#     book_db = FAISS.load_local(DB_FAISS_PATH_DATA, embeddings, allow_dangerous_deserialization=True)
#     qp_db = FAISS.load_local(DB_FAISS_PATH_QP, embeddings, allow_dangerous_deserialization=True)

#     # Load data from FAISS
#     book_data = book_db.similarity_search("extract relevant book data")[0].page_content
#     qp_structure = qp_db.similarity_search("extract relevant question paper structure")[0].page_content
#     return book_data, qp_structure




# def gen_qp():
#     # Define folder to save the question paper
#     output_folder = "generated_question_papers"
    
#     # Create the folder if it doesn't exist
#     os.makedirs(output_folder, exist_ok=True)
    
#     # Generate timestamped filename
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     file_name = f"question_paper_{timestamp}.md"
#     file_path = os.path.join(output_folder, file_name)
    
#     # Load embeddings
#     print("Loading embeddings...")
#     try:
#         book_data, qp_structure = load_embeddings()
#     except Exception as e:
#         print(f"Error during embeddings loading: {e}")
#         return
    
#     # Generate question paper
#     print("Generating question paper...")
#     try:
#         question_paper = generate_question_paper(book_data, qp_structure)
#     except Exception as e:
#         print(f"Error during question paper generation: {e}")
#         return
    
#     # Save the question paper
#     print("Saving question paper...")
#     try:
#         with open(file_path, "w", encoding='utf-8') as file:
#             file.write(question_paper)
#         print(f"Question paper generated and saved successfully at {file_path}!")
#         return file_path
#     except Exception as e:
#         print(f"Failed to save question paper: {e}")

# file_path = gen_qp()        

# from ans_gen import generate_answer

# # Generate and save answers
# answers_path = generate_answer(file_path) # So it will automatically consider the latest question paper generated
# print("Answers saved at:", answers_path)


from langchain.prompts import PromptTemplate
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from datetime import datetime
import os
import random

# ===============================
# ENV & CONFIG
# ===============================
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv('API_KEY')
genai.configure(api_key=os.getenv("API_KEY"))

# Path of vector databases
DB_FAISS_PATH_DATA = 'vectorstore_data/db_faiss'
DB_FAISS_PATH_QP = 'vectorstore_Syllabus/db_faiss'

"""
using random.uniform(0.5,1) ---> for maintaining randomness of the question papers.
changed model to gemini-2.0-flash
"""

# ===============================
# LLM LOADER
# ===============================
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=random.uniform(0.5, 1)
    )

# ===============================
# PROMPT TEMPLATE
# ===============================
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


4. Question Format Example:
   Q.2 Solve the following:
   A) Explain the concept of requirements engineering and its importance in software development. (6)
   B) Discuss various requirements elicitation techniques with examples. (6)

   Q.4 Solve any TWO of the following:
   A) What is system modeling? Explain its significance in software development. (6)
   B) Describe the different types of UML diagrams with their purposes. (6)
   C) Explain the concept of behavioral modeling with suitable examples. (6)

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

Use this structure for the question paper:

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

Context for question generation (from book):
{context}

Syllabus structure:
{qp_structure}

Generate a complete question paper following all the above requirements strictly, ensuring each question/sub-question ends with marks in parentheses.
"""

question_paper_prompt = PromptTemplate(
    template=question_paper_template,
    input_variables=['context', 'qp_structure']
)

# ===============================
# EMBEDDINGS / FAISS LOADING
# ===============================
def load_embeddings():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    book_db = FAISS.load_local(DB_FAISS_PATH_DATA, embeddings, allow_dangerous_deserialization=True)
    qp_db = FAISS.load_local(DB_FAISS_PATH_QP, embeddings, allow_dangerous_deserialization=True)

    # Load data from FAISS
    book_data_doc = book_db.similarity_search("extract relevant book data")[0]
    qp_structure_doc = qp_db.similarity_search("extract relevant question paper structure")[0]

    book_data = book_data_doc.page_content
    qp_structure = qp_structure_doc.page_content

    return book_data, qp_structure

# ===============================
# QUESTION PAPER GENERATION
# ===============================
def generate_question_paper(book_data: str, qp_structure: str) -> str:
    """
    Calls the LLM with a formatted prompt and returns the generated question paper as a string.
    """
    llm = load_llm()
    full_prompt = question_paper_prompt.format(
        context=book_data,
        qp_structure=qp_structure
    )
    response = llm.invoke(full_prompt)

    # ChatGoogleGenerativeAI returns an AIMessage; content is the text.
    return response.content if hasattr(response, "content") else str(response)

# ===============================
# MAIN GENERATOR
# ===============================
def gen_qp():
    # Define folder to save the question paper
    output_folder = "generated_question_papers"
    os.makedirs(output_folder, exist_ok=True)

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"question_paper_{timestamp}.md"
    file_path = os.path.join(output_folder, file_name)

    # Load embeddings
    print("Loading embeddings...")
    book_data, qp_structure = load_embeddings()

    # Generate question paper
    print("Generating question paper...")
    question_paper = generate_question_paper(book_data, qp_structure)

    if not isinstance(question_paper, str) or len(question_paper.strip()) == 0:
        raise ValueError("Generated question paper is empty or not a string.")

    # Save the question paper
    print("Saving question paper...")
    with open(file_path, "w", encoding='utf-8') as file:
        file.write(question_paper)

    print(f"Question paper generated and saved successfully at {file_path}!")
    return file_path


# ===============================
# ENTRY POINT (QP + ANSWERS)
# ===============================
if __name__ == "__main__":
    from ans_gen import generate_answer

    qp_path = gen_qp()          # generate question paper & get path

    if not qp_path or not os.path.isfile(qp_path):
        raise RuntimeError("Question paper generation failed; cannot generate answers.")

    answers_path = generate_answer(qp_path)  # pass a valid .md path
    print("Answers saved at:", answers_path)
