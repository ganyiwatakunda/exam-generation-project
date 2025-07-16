# app.py
import os
import re
import streamlit as st
import pandas as pd
from io import BytesIO
from fpdf import FPDF

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

import evaluation  # our evaluation module

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

# PDF generator class
class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "Generated Exam Paper", ln=True, align="C")
        self.ln(10)

    def chapter_body(self, content):
        self.set_font("Arial", "", 12)
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

# Core generate exam function (RAG + LLM)
def generate_exam_response(role: str, subject: str, paper_type: str, prompt: str) -> str:
    if not prompt or subject not in VALID_SUBJECTS or role not in VALID_ROLES:
        raise ValueError("Invalid role, subject, or prompt")

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    docs = load_documents(subject, paper_type)
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Simple prompt template for demo (you can expand based on subject/paper)
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
    prompt_template = PromptTemplate(template=template, input_variables=["context", "question"])

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    retrieved_docs = retriever.get_relevant_documents(prompt)

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

# Generate LLM only response (no RAG)
def generate_llm_only_response(prompt):
    llm = ChatOpenAI(model_name=MODEL_NAME)
    return llm.invoke(prompt)

# RAG + LLM without textbooks
def generate_rag_llm_no_textbook(role, subject, paper_type, prompt):
    def custom_load_documents(subject, paper_type):
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
        # OMIT textbook folder here intentionally (exclude textbooks)

        docs = []
        for folder in selected_folders:
            full_path = os.path.join(subject_path, folder)
            if os.path.exists(full_path):
                loader = DirectoryLoader(full_path)
                docs.extend(loader.load())
        return docs

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    docs = custom_load_documents(subject, paper_type)
    vectorstore = FAISS.from_documents(docs, embeddings)

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
    prompt_template = PromptTemplate(template=template, input_variables=["context", "question"])

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    retrieved_docs = retriever.get_relevant_documents(prompt)

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

# === Streamlit UI ===
st.set_page_config(page_title="Exam Generator Chatbot with Evaluation", layout="wide")
st.title("📘 Exam Generation Bot for Zimsec Grade 7 Subjects with Evaluation")

with st.sidebar:
    st.header("📚 User Guide")
    st.markdown("""
- **Step 1**: Select your role (Teacher or Student)  
- **Step 2**: Choose a subject  
- **Step 3**: Select Paper 1 or Paper 2  
- **Step 4**: Accept or modify the prompt  
- **Step 5**: Click **Generate Exam**  
- **Step 6**: View evaluation metrics for different generation scenarios  
""")

role = st.selectbox("🎓 Select your role", ["Select role"] + VALID_ROLES)
subject = st.selectbox("📖 Select subject", ["Select subject"] + VALID_SUBJECTS)
paper_type = st.selectbox("📄 Select paper type", ["Paper 1", "Paper 2"])

default_prompt = f"Generate a comprehensive Zimbabwe Grade 7 {subject} {paper_type} exam paper with questions and marking scheme."
prompt = st.text_area("✍️ Prompt (modify if you want)", default_prompt, height=150)

if role != "Select role" and subject != "Select subject" and paper_type and prompt.strip():
    if st.button("🚀 Generate Exam"):

        with st.spinner("Generating exam papers and evaluating..."):
            try:
                # Scenario 1: LLM only (no RAG)
                output_llm_only = generate_llm_only_response(prompt)
                scores_llm_only = evaluation.evaluate_exam_paper(output_llm_only)

                # Scenario 2: RAG + LLM (full, with textbooks)
                output_rag_llm = generate_exam_response(role, subject, paper_type, prompt)
                scores_rag_llm = evaluation.evaluate_exam_paper(output_rag_llm)

                # Scenario 3: RAG + LLM without textbooks
                output_rag_llm_no_text = generate_rag_llm_no_textbook(role, subject, paper_type, prompt)
                scores_rag_llm_no_text = evaluation.evaluate_exam_paper(output_rag_llm_no_text)

                # Show evaluation metrics in table
                df_scores = pd.DataFrame([
                    scores_llm_only,
                    scores_rag_llm,
                    scores_rag_llm_no_text
                ], index=["LLM only", "RAG + LLM", "RAG + LLM w/o Textbook"])

                st.subheader("⚖️ Evaluation Metrics Comparison")
                st.dataframe(df_scores.style.highlight_max(axis=0))

                # Show the generated exam for RAG + LLM full
                st.subheader("📄 Generated Exam (RAG + LLM Full)")
                st.code(output_rag_llm[:1000] + "...")  # Show first 1000 chars

                # PDF download
                pdf_buffer = BytesIO()
                pdf = PDF()
                pdf.add_page_with_content(output_rag_llm)
                pdf.output(pdf_buffer)
                pdf_buffer.seek(0)

                st.download_button(
                    label="📥 Download Exam as PDF",
                    data=pdf_buffer,
                    file_name=f"Grade7_{subject.replace(' ','_')}_{paper_type.replace(' ', '')}_Exam.pdf",
                    mime="application/pdf",
                )

            except Exception as e:
                st.error(f"Error generating exams: {e}")
else:
    st.info("Please select role, subject, paper type, and enter a prompt.")
