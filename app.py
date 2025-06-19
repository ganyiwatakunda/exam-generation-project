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
        self.multi_cell(0, 10, content)

    def add_page_with_content(self, content):
        self.add_page()
        self.chapter_body(content)

def generate_exam_response(role: str, subject: str, paper_type: str, prompt: str) -> str:
    if not prompt or subject not in VALID_SUBJECTS or role not in VALID_ROLES:
        raise ValueError("Invalid role, subject, or prompt")

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    docs = load_documents(subject, paper_type)
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Define templates depending on role, subject, paper_type
    if role == "Teacher":
        if subject == "Social Science" and paper_type == "Paper 1":
            template = f"""
You are an exam generator for the Zimbabwe Grade 7 {subject} subject.

Use the context below to ensure curriculum relevance:
```{{context}}```

Prompt: {{question}}

Generate a full Paper 1 exam:
- Title: Grade 7 {subject} Examination - Paper 1
- ===INSTRUCTIONS===
Provide detailed candidate instructions for the exam.
- ===QUESTIONS===
Exactly 40 Multiple Choice Questions, numbered 1 to 40
Each question has 4 choices (A, B, C, D)
Include at least 5 diagram or map based questions (e.g. [Insert diagram of river system], [Insert map of Zimbabwe])
- ===ANSWER KEY===
Provide answer key in the format "1: B, 2: D, ..." with clear marking scheme
"""
        elif subject == "Social Science" and paper_type == "Paper 2":
            template = f"""
You are an exam generator for the Zimbabwe Grade 7 {subject} subject.

Use the context below to ensure curriculum relevance:
```{{context}}```

Prompt: {{question}}

Generate a full Paper 2 exam:
- Title: Grade 7 {subject} Examination - Paper 2
- ===INSTRUCTIONS===
Provide candidate instructions including section details (Section A, B, C) and exam rules.
- ===QUESTIONS===
Structured into Section A, Section B and Section C
Include marks allocation per question and diagrams/maps where appropriate
- ===ANSWER KEY===
Provide a detailed marking scheme with model answers
"""
        else:
            # Default template for other subjects/papers
            template = f"""
You are an exam generator for the Zimbabwe Grade 7 {subject} subject.

Use the context below to ensure curriculum relevance:
```{{context}}```

Prompt: {{question}}

Generate a full exam:
- Title: Grade 7 {subject} Examination - {paper_type}
- ===INSTRUCTIONS===
Provide candidate instructions appropriate for the subject and exam type.
- ===QUESTIONS===
Provide exam questions with proper formatting and marks allocation.
- ===ANSWER KEY===
Provide detailed answers or marking scheme.
"""
    else:
        # Student template — no answers included
        template = f"""
You are a revision paper generator for Grade 7 students in Zimbabwe studying {subject}.

Use the context below to ensure curriculum relevance:
```{{context}}```

Prompt: {{question}}

Generate a mock {paper_type} revision exam paper:
- Title: Grade 7 {subject} Practice Questions - {paper_type}
- ===INSTRUCTIONS===
Provide candidate instructions for practice.
- ===QUESTIONS===
Provide exam-style questions only; do NOT include answers.
"""

    prompt_template = PromptTemplate(template=template, input_variables=["context", "question"])

    retriever = vectorstore.as_retriever()
    retrieved_docs = retriever.get_relevant_documents(prompt)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    llm = ChatOpenAI(model_name=MODEL_NAME)
    chain = (
        {"context": lambda _: context, "question": RunnablePassthrough()} |
        prompt_template |
        llm |
        StrOutputParser()
    )

    return chain.invoke(prompt)

# Helper to split exam output into sections
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
            # fallback if markers missing
            questions = exam_text
    except Exception:
        # fail safe fallback
        questions = exam_text

    return instructions, questions, answers

# === Streamlit App ===
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
- **Step 6**: Download the exam paper(s) and marking scheme (if teacher)  
""")

    st.markdown("""
---
### ℹ️ About
This app uses generative AI with curriculum-aligned textbooks and papers to generate Grade 7 exam papers for Zimbabwe.
""")

role = st.selectbox("🎓 Select your role", ["Select"] + VALID_ROLES)
subject = st.selectbox("📘 Select Subject", ["Select"] + VALID_SUBJECTS)

if subject != "Select" and role != "Select":
    paper_type = st.radio("🧾 Select Exam Type", ["Paper 1", "Paper 2"])
    pre_prompt = f"Create a {subject} {paper_type} exam"
    prompt = st.text_area("✏️ Prompt", value=pre_prompt)

    if st.button("🚀 Generate Exam") and prompt:
        with st.spinner("Generating exam paper..."):
            try:
                output = generate_exam_response(role, subject, paper_type, prompt)
                instructions, questions, answers = split_exam_sections(output)

                if role == "Teacher":
                    st.subheader("📄 Candidate Instructions + Questions")
                    st.code(f"{instructions}\n\n{questions}")

                    st.subheader("📝 Marking Scheme (Answers)")
                    st.code(answers)

                    # Create Instructions + Questions PDF (instructions page + questions page)
                    pdf_ij = PDF()
                    pdf_ij.add_page()
                    pdf_ij.chapter_body(instructions)  # Instructions page

                    pdf_ij.add_page()
                    pdf_ij.chapter_body(questions)     # Questions start on new page

                    pdf_ij_output = pdf_ij.output(dest='S').encode('latin1')
                    pdf_ij_buffer = BytesIO(pdf_ij_output)

                    st.download_button(
                        label="⬇️ Download Instructions + Questions (PDF)",
                        data=pdf_ij_buffer,
                        file_name=f"{subject}_{paper_type}_instructions_questions.pdf",
                        mime="application/pdf"
                    )

                    # Create Marking Scheme PDF
                    pdf_answers = PDF()
                    pdf_answers.add_page()
                    pdf_answers.chapter_body("Marking Scheme\n\n" + answers)

                    pdf_answers_output = pdf_answers.output(dest='S').encode('latin1')
                    pdf_answers_buffer = BytesIO(pdf_answers_output)

                    st.download_button(
                        label="⬇️ Download Marking Scheme (PDF)",
                        data=pdf_answers_buffer,
                        file_name=f"{subject}_{paper_type}_marking_scheme.pdf",
                        mime="application/pdf"
                    )

                    # TXT Downloads for Teachers
                    # Instructions + Questions TXT
                    ij_txt = f"{instructions}\n\n{questions}"
                    ij_txt_buffer = BytesIO(ij_txt.encode("utf-8"))
                    st.download_button(
                        label="⬇️ Download Instructions + Questions (.txt)",
                        data=ij_txt_buffer,
                        file_name=f"{subject}_{paper_type}_instructions_questions.txt",
                        mime="text/plain"
                    )

                    # Marking Scheme TXT
                    ans_txt = "Marking Scheme\n\n" + answers
                    ans_txt_buffer = BytesIO(ans_txt.encode("utf-8"))
                    st.download_button(
                        label="⬇️ Download Marking Scheme (.txt)",
                        data=ans_txt_buffer,
                        file_name=f"{subject}_{paper_type}_marking_scheme.txt",
                        mime="text/plain"
                    )

                else:  # Student
                    st.subheader("📄 Practice Questions (No Answers)")
                    st.code(output)

                    # PDF download for students (practice questions only)
                    pdf_practice = PDF()
                    pdf_practice.add_page()
                    pdf_practice.chapter_body(output)
                    pdf_practice_output = pdf_practice.output(dest='S').encode('latin1')
                    pdf_practice_buffer = BytesIO(pdf_practice_output)
                    st.download_button(
                        label="⬇️ Download Practice Questions (PDF)",
                        data=pdf_practice_buffer,
                        file_name=f"{subject}_{paper_type}_practice_questions.pdf",
                        mime="application/pdf"
                    )

                    # TXT download for students
                    practice_buffer = BytesIO(output.encode("utf-8"))
                    st.download_button(
                        label="⬇️ Download Practice Questions (.txt)",
                        data=practice_buffer,
                        file_name=f"{subject}_{paper_type}_practice_questions.txt",
                        mime="text/plain"
                    )
            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.info("Please select your role and subject.")
