# Streamlit Integrated Exam Generator for Grade 7 (RAG + OpenAI)
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

import os
import dotenv
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

    def add_exam(self, content):
        self.add_page()
        self.chapter_body(content)

def generate_exam_response(role: str, subject: str, paper_type: str, prompt: str) -> str:
    if not prompt or subject not in VALID_SUBJECTS or role not in VALID_ROLES:
        raise ValueError("Invalid role, subject, or prompt")

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    docs = load_documents(subject, paper_type)
    vectorstore = FAISS.from_documents(docs, embeddings)

    if role == "Teacher":
        if subject == "Social Science" and paper_type == "Paper 1":
            template = f"""
            You are an exam generator for the Zimbabwe Grade 7 {subject} subject.
            
            Use the context below to ensure curriculum relevance:
            ```{{context}}```
            
            Prompt: {{question}}
            
            Generate a full Paper 1 exam:
            - Title: Grade 7 {subject} Examination - Paper 1
            - Instructions to candidates
            - Exactly 40 Multiple Choice Questions (MCQs)
            - Numbered from 1 to 40
            - 4 options per question: A, B, C, D
            - Include at least 5 questions that reference diagrams or maps about families or cultures as in the standard paper formats in the rsources folder 
            - Provide answer key at the end (e.g., 1: B, 2: D, ...)
            - Ensure the questions reflect actual past exam tone and topic coverage
            """

            template = f"""
            You are an exam generator for the Zimbabwe Grade 7 {subject} subject.

            Use the context below to ensure curriculum relevance:
            ```{{context}}```

            Prompt: {{question}}

            Generate a full Paper 1 exam:
            - Title: Grade 7 {subject} Examination - Paper 1
            - Instructions to candidates
            - 40 Multiple Choice Questions
            - Numbered 1 to 40
            - 4 choices per question (A, B, C, D)
            - Include diagrams or images where applicable
            - Provide sample answers at the end
            """
        elif subject == "Social Science" and paper_type == "Paper 2":
            template = f"""
            You are an exam generator for the Zimbabwe Grade 7 {subject} subject.

            Use the context below to ensure curriculum relevance:
            ```{{context}}```

            Prompt: {{question}}

            Generate a full Paper 2 exam:
            - Title: Grade 7 {subject} Examination - Paper 2
            - Candidate Instructions
            - Structured into Section A, Section B and Section C
            - Use correct section formatting based on past papers
            - Include diagrams or maps where appropriate
            - Each question should indicate marks
            - Provide sample answers at the end
            """
        else:
            template = f"""
            You are an exam generator for the Zimbabwe Grade 7 {subject} subject.

            Use the context below to ensure curriculum relevance:
            ```{{context}}```

            Prompt: {{question}}

            Generate a full exam:
            - Title: Grade 7 {subject} Examination - {paper_type}
            - Structured appropriately per subject norms (skip sections for Paper 1 if not used)
            - Include instructions, clear formatting, and mark allocations
            - Include diagrams or visual aids if relevant
            - Provide sample answers at the end
            """
    else:
        template = f"""
        You are a revision paper generator for Grade 7 students in Zimbabwe studying {subject}.

        Use the context below to ensure curriculum relevance:
        ```{{context}}```

        Prompt: {{question}}

        Generate a mock {paper_type} revision exam paper:
        - Title: Grade 7 {subject} Revision Paper - {paper_type}
        - Candidate instructions
        - Full set of exam-style questions
        - Indicate marks per question
        - Exclude answers to encourage practice
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

# === Streamlit App ===
st.set_page_config(page_title="Exam Generator Chatbot", layout="centered")
st.title("📘 Exam Generation Bot for Zimsec Grade 7 Subjects")

st.markdown("""
### 📝 Overview
Welcome to the Grade 7 Exam Generator!

This system uses AI and educational materials to generate mock and real exam papers for Zimbabwe's Grade 7 curriculum.

- 📚 **Curriculum-Based**: Generates exams based on uploaded past papers and textbooks.
- 🎯 **Subjects Supported**: Social Science, English, Mathematics, Science & Tech, Agriculture Science & Tech.
- ✍️ **Paper Format Matching**: Matches real exam formats including multiple choice and sectioned structured exams.
- 📥 **Downloadable**: Exams can be downloaded as `.txt` or `.pdf` files.

### 🧑‍🏫 How to Use
1. **Select your role**: Teacher or Student
2. **Choose a subject**
3. **Pick Paper 1 or Paper 2** (Paper 1 = multiple choice, Paper 2 = structured questions)
4. **Modify or accept pre-filled prompt**
5. **Click 'Generate Exam'**
6. **Download and review**
""")

role = st.selectbox("Select your role", ["Select"] + VALID_ROLES)
subject = st.selectbox("Select Subject", ["Select"] + VALID_SUBJECTS)

if subject != "Select" and role != "Select":
    paper_type = st.radio("Select Exam Type", ["Paper 1", "Paper 2"])
    pre_prompt = f"Create a {subject} {paper_type} exam"
    prompt = st.text_area("Prompt", value=pre_prompt)

    if st.button("Generate Exam") and prompt:
        with st.spinner("Generating exam paper..."):
            try:
                output = generate_exam_response(role, subject, paper_type, prompt)
                st.subheader("📄 Generated Exam Paper")
                st.code(output)

                # Save as TXT
                txt_buffer = BytesIO()
                txt_buffer.write(output.encode("utf-8"))
                txt_buffer.seek(0)

                st.download_button(
                    label="⬇️ Download as .txt",
                    data=txt_buffer,
                    file_name=f"{subject}_{paper_type}.txt",
                    mime="text/plain"
                )

                # Save as PDF
                import unicodedata

                # Clean the output for PDF compatibility
                clean_output = unicodedata.normalize("NFKD", output).encode("ascii", "ignore").decode("ascii")
                
                pdf.add_exam(clean_output)
                pdf_output = pdf.output(dest='S').encode('latin1')
                pdf_buffer = BytesIO(pdf_output)

                
               # pdf = PDF()
                #pdf.add_exam(output)
                #pdf_output = pdf.output(dest='S').encode('latin1')
                #pdf_buffer = BytesIO(pdf_output)
                
                st.download_button(
                    label="⬇️ Download as PDF",
                    data=pdf_buffer,
                    file_name=f"{subject}_{paper_type}.pdf",
                    mime="application/pdf"
                )

            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.info("Please select your role and subject.")
