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

# Exam generation logic with truncation

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
        elif subject == "English" and paper_type == "Paper 1":
            template = f"""
You are an exam generator for the Zimbabwe Grade 7 English subject.

Use the context below to ensure curriculum relevance:
```{{context}}```

Prompt: {{question}}

Generate a full English Paper 1 exam:
- Title: Grade 7 English Examination - Paper 1
- ===INSTRUCTIONS===
Provide candidate instructions.
- ===QUESTIONS===
Generate a paper with the following structure:
  - At least 4 comprehension passages, each followed by at least 6 questions
  - Interleave language-based sections after each passage with about 8 questions
  - Continue alternating until a total of 50 questions is reached
- ===ANSWER KEY===
Provide a detailed marking scheme
"""
        elif subject == "English" and paper_type == "Paper 2":
            template = f"""
You are an exam generator for the Zimbabwe Grade 7 English subject.

Use the context below to ensure curriculum relevance:
```{{context}}```

Prompt: {{question}}

Generate a full English Paper 2 exam:
- Title: Grade 7 English Examination - Paper 2
- ===INSTRUCTIONS===
Provide candidate instructions.
- ===QUESTIONS===
Section A: Composition or letter writing (20 marks) — provide guidelines and options
Section B: One comprehension passage with questions totaling 15 marks
- ===ANSWER KEY===
Provide a detailed marking scheme and model answers
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
Provide candidate instructions appropriate for the subject and exam type.
- ===QUESTIONS===
Provide exam questions with proper formatting and marks allocation.
- ===ANSWER KEY===
Provide detailed answers or marking scheme.
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
Provide candidate instructions for practice.
- ===QUESTIONS===
Provide exam-style questions only; do NOT include answers.
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
