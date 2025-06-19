# Streamlit Integrated Exam Generator for Grade 7 (RAG + OpenAI)
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

import os
import streamlit as st
from io import BytesIO
from fpdf import FPDF
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

# === CONFIG ===
BASE_RESOURCE_PATH = "./resources"
STORAGE_PATH = "./vectorstore"
EMBEDDING_MODEL = "text-embedding-3-large"
MODEL_NAME = "gpt-3.5-turbo"

VALID_SUBJECTS = ["Mathematics", "English", "Agriculture Science and Technology", "Science and Technology", "Social Science"]
VALID_ROLES = ["Student", "Teacher"]

# Function to load subject documents based on paper type
def load_documents(subject, paper_type):
    subject_folder_map = {
        "Agriculture Science and Technology": "agriculturescienceandtechnology",
        "Science and Technology": "scienceandtechnology",
        "Social Science": "socialscience",
        "Mathematics": "mathematics",
        "English": "english"
    }
    subject_path = os.path.join(BASE_RESOURCE_PATH, subject_folder_map.get(subject, subject.lower().replace(" ", "")))
    selected_folders = []

    if subject == "Social Science":
        if paper_type == "Paper 1":
            selected_folders.append("paper1")
        elif paper_type == "Paper 2":
            selected_folders.append("paper2")
    selected_folders.append("textbook")

    docs = []
    for folder in selected_folders:
        full_path = os.path.join(subject_path, folder)
        if os.path.exists(full_path):
            loader = DirectoryLoader(full_path)
            docs.extend(loader.load())
    return docs

# PDF generator
class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "Generated Exam Paper", ln=True, align="C")
        self.ln(10)

    def chapter_body(self, content):
        self.set_font("Arial", "", 12)
        #self.multi_cell(0, 10, content)
        safe_content = (
        content.replace("’", "'")
               .replace("“", '"')
               .replace("”", '"')
               .replace("–", "-")
               .replace("—", "-")
               .replace("•", "*")
               .replace("…", "...")
    )
    self.multi_cell(0, 10, safe_content)

    def add_page_with_content(self, content):
        self.add_page()
        self.chapter_body(content)

# Exam generation logic with truncation

def generate_exam_response(role: str, subject: str, paper_type: str, prompt: str) -> str:
    if not prompt or subject not in VALID_SUBJECTS or role not in VALID_ROLES:
        raise ValueError("Invalid role, subject, or prompt")

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    docs = load_documents(subject, paper_type)
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Prompt templates for different subjects and paper types
    if role == "Teacher":
        if subject == "English" and paper_type == "Paper 1":
            template = """
You are an exam generator for the Zimbabwe Grade 7 English subject.

Use the context below to ensure curriculum relevance:
```{context}```

Prompt: {question}

Generate a full English Paper 1 exam:
- Title: Grade 7 English Examination - Paper 1
- ===INSTRUCTIONS===
Include clear instructions for the candidates to answer all 50 questions.
- ===QUESTIONS===
Structure:
    - Include at least 4 comprehension passages
    - Each followed by a minimum of 6 questions
    - Alternate each passage with 8 language-based questions (fill in blanks, punctuation, sentence correction)
    - Continue until total of 50 questions is reached
- ===ANSWER KEY===
Provide answers in format: 1: A, 2: B, ..., 50: D
"""

        elif subject == "English" and paper_type == "Paper 2":
            template = """
You are an exam generator for the Zimbabwe Grade 7 English subject.

Use the context below to ensure curriculum relevance:
```{context}```

Prompt: {question}

Generate a full English Paper 2 exam:
- Title: Grade 7 English Examination - Paper 2
- ===INSTRUCTIONS===
Provide clear instructions for both sections.
- ===QUESTIONS===
Section A (20 marks):
    - Letter or composition writing
    - Include prompts or guidelines
Section B (15 marks):
    - One comprehension passage
    - Include questions totalling 15 marks
- ===ANSWER KEY===
Give a simple rubric for Section A and detailed answers for Section B.
"""

        elif subject == "Social Science" and paper_type == "Paper 1":
            template = """
You are an exam generator for the Zimbabwe Grade 7 Social Science subject.

Use the context below to ensure curriculum relevance:
```{context}```

Prompt: {question}

Generate a full Paper 1 exam:
- Title: Grade 7 Social Science Examination - Paper 1
- ===INSTRUCTIONS===
Provide candidate instructions.
- ===QUESTIONS===
40 Multiple Choice Questions (A–D), include at least 5 diagram/map-based questions
- ===ANSWER KEY===
Format: 1: C, 2: A, ..., 40: B
"""

        elif subject == "Social Science" and paper_type == "Paper 2":
            template = """
You are an exam generator for the Zimbabwe Grade 7 Social Science subject.

Use the context below to ensure curriculum relevance:
```{context}```

Prompt: {question}

Generate a full Paper 2 exam:
- Title: Grade 7 Social Science Examination - Paper 2
- ===INSTRUCTIONS===
Give instructions and describe sections.
- ===QUESTIONS===
Include Section A, B, and C with marks and diagrams/maps where needed
- ===ANSWER KEY===
Detailed answers with mark allocation.
"""

        elif subject == "Agriculture Science and Technology" and paper_type == "Paper 1":
            template = """
You are an exam generator for the Zimbabwe Grade 7 Agriculture Science and Technology subject.

Use the context below to ensure curriculum relevance:
```{context}```

Prompt: {question}

Generate a full Paper 1 exam:
- Title: Grade 7 Agriculture Science and Technology Examination - Paper 1
- ===INSTRUCTIONS===
Candidates must answer all 50 multiple choice questions.
- ===QUESTIONS===
Provide 50 multiple choice questions (A–D options)
- ===ANSWER KEY===
List answers in format: 1: A, 2: B, ..., 50: C
"""

        elif subject == "Agriculture Science and Technology" and paper_type == "Paper 2":
            template = """
You are an exam generator for the Zimbabwe Grade 7 Agriculture Science and Technology subject.

Use the context below to ensure curriculum relevance:
```{context}```

Prompt: {question}

Generate a full Paper 2 exam:
- Title: Grade 7 Agriculture Science and Technology Examination - Paper 2
- ===INSTRUCTIONS===
Candidates must answer all 5 questions from Section A and 1 question from each of the remaining three sections (B, C, D)
- ===QUESTIONS===
Questions that are generated must have sub questions also and the marks allocation per question must have a breakdown
Section A: 5 questions
Section B: 3 questions
Section C: 3 questions
Section D: 3 questions
- ===ANSWER KEY===
Detailed answers with marks per question.
"""

        else:
            template = f"""
You are an exam generator for the Zimbabwe Grade 7 {subject} subject.

Use the context below to ensure curriculum relevance:
```{{context}}```

Prompt: {{question}}

Generate a full exam:
- Title: Grade 7 {subject} Examination - {paper_type}
- ===INSTRUCTIONS===
Provide candidate instructions.
- ===QUESTIONS===
Include clear formatting, question numbers, and marks
- ===ANSWER KEY===
Provide correct answers with marking guidance.
"""

    else:
        template = f"""
You are a revision paper generator for Grade 7 students in Zimbabwe studying {subject}.

Use the context below to ensure curriculum relevance:
```{{context}}```

Prompt: {{question}}

Generate a mock {paper_type} revision exam paper:
- Title: Grade 7 {subject} Practice Questions - {paper_type}
- ===INSTRUCTIONS===
Give practice instructions
- ===QUESTIONS===
Format like real exam. DO NOT include answers.
"""

    prompt_template = PromptTemplate(template=template, input_variables=["context", "question"])

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    retrieved_docs = retriever.get_relevant_documents(prompt)

    # === CONTEXT TRUNCATION ===
    max_chars = 45000
    combined_context = ""
    for doc in retrieved_docs:
        if len(combined_context) + len(doc.page_content) <= max_chars:
            combined_context += doc.page_content + "\n\n"
        else:
            break

    context = combined_context

    llm = ChatOpenAI(model_name=MODEL_NAME)
    chain = (
        {"context": lambda _: context, "question": RunnablePassthrough()} |
        prompt_template |
        llm |
        StrOutputParser()
    )

    return chain.invoke(prompt)

def split_exam_sections(exam_text):
    instructions = ""
    questions = ""
    answers = ""

    try:
        if "===INSTRUCTIONS===" in exam_text and "===QUESTIONS===" in exam_text:
            parts = exam_text.split("===INSTRUCTIONS===")[1].split("===QUESTIONS===")
            instructions = parts[0].strip()
            rest = parts[1].strip()

            if "===ANSWER KEY===" in rest:
                questions, answers = rest.split("===ANSWER KEY===")
                questions = questions.strip()
                answers = answers.strip()
            else:
                questions = rest.strip()
        else:
            questions = exam_text
    except Exception:
        questions = exam_text

    return instructions, questions, answers

# === Streamlit UI ===
st.set_page_config(page_title="Exam Generator Chatbot", layout="wide")
st.title("📘 Exam Generation Bot for Zimsec Grade 7 Subjects")

with st.sidebar:
    st.header("📚 User Guide")
    st.markdown("""
- **Step 1**: Select your role (Teacher or Student)  
- **Step 2**: Choose a subject  
- **Step 3**: Select Paper 1 or Paper 2  
- **Step 4**: Accept or modify the prompt  
- **Step 5**: Click **Generate Exam**  
- **Step 6**: Download your generated paper  
""")

role = st.selectbox("🎓 Select your role", ["Select"] + VALID_ROLES)
subject = st.selectbox("📘 Select Subject", ["Select"] + VALID_SUBJECTS)

if subject != "Select" and role != "Select":
    paper_type = st.radio("\U0001F9FE Select Exam Type", ["Paper 1", "Paper 2"])
    default_prompt = f"Create a {subject} {paper_type} exam"
    prompt = st.text_area("✏️ Prompt", value=default_prompt)

    if st.button("\U0001F680 Generate Exam") and prompt:
        with st.spinner("Generating exam paper..."):
            try:
                output = generate_exam_response(role, subject, paper_type, prompt)

                if "===ANSWER KEY===" in output:
                    instructions, rest = output.split("===QUESTIONS===", 1)
                    questions, answers = rest.split("===ANSWER KEY===", 1)
                else:
                    instructions, questions, answers = output, "", ""

                # Prepare PDFs
                pdf_files = {}

                if role == "Teacher":
                    st.subheader("\U0001F4C4 Candidate Instructions + Questions")
                    st.code(f"{instructions}\n\n{questions}")

                    st.subheader("\U0001F4DD Marking Scheme (Answers)")
                    st.code(answers)

                    pdf_ij = PDF()
                    pdf_ij.add_page()
                    pdf_ij.chapter_body(instructions)
                    pdf_ij.add_page()
                    pdf_ij.chapter_body(questions)
                    pdf_ij_buffer = BytesIO(pdf_ij.output(dest='S').encode('latin1'))
                    pdf_files["Instructions + Questions"] = pdf_ij_buffer

                    pdf_ans = PDF()
                    pdf_ans.add_page()
                    pdf_ans.chapter_body("Marking Scheme\n\n" + answers)
                    pdf_ans_buffer = BytesIO(pdf_ans.output(dest='S').encode('latin1'))
                    pdf_files["Marking Scheme"] = pdf_ans_buffer

                else:
                    st.subheader("\U0001F4C4 Practice Questions (No Answers)")
                    st.code(output)

                    pdf_practice = PDF()
                    pdf_practice.add_page()
                    pdf_practice.chapter_body(output)
                    pdf_practice_buffer = BytesIO(pdf_practice.output(dest='S').encode('latin1'))
                    pdf_files["Practice Questions"] = pdf_practice_buffer

                # Show all download buttons
                for label, buffer in pdf_files.items():
                    st.download_button(f"⬇️ Download {label} (PDF)", data=buffer,
                                       file_name=f"{subject}_{paper_type}_{label.lower().replace(' ', '_')}.pdf",
                                       mime="application/pdf")

            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.info("Please select your role and subject.")
