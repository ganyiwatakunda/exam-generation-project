# Streamlit Integrated Exam Generator for Grade 7 (RAG + OpenAI)

import os
import dotenv
import streamlit as st
from io import BytesIO
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

# === CONFIG ===
RESOURCE_PATH = "./resources"
STORAGE_PATH = "./vectorstore"
EMBEDDING_MODEL = "text-embedding-3-large"
MODEL_NAME = "gpt-3.5-turbo"

#dotenv.load_dotenv(".env")

VALID_SUBJECTS = ["Mathematics", "English", "ICT", "Social Science"]
VALID_ROLES = ["Student", "Teacher"]

def generate_exam_response(role: str, subject: str, prompt: str) -> str:
    if not prompt or subject not in VALID_SUBJECTS or role not in VALID_ROLES:
        raise ValueError("Invalid role, subject, or prompt")

    from openai import OpenAI
    #embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    client = OpenAI()  # no proxies passed
    #embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL,client=OpenAI())  # ensures compatible instantiation with no unexpected proxies )
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL,client=client)  # ensures compatible instantiation with no unexpected proxies )

    if os.path.exists(STORAGE_PATH):
        vectorstore = FAISS.load_local(STORAGE_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        loader = DirectoryLoader(RESOURCE_PATH)
        docs = loader.load()
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(STORAGE_PATH)

    if role == "Teacher":
        template = f"""
        You are an exam paper generator for the Zimbabwe Grade 7 {subject} subject.

        Use the context below to ensure curriculum relevance:
        ```{{context}}```

        Prompt: {{question}}

        You must output a complete formal exam paper:
        - Title: Grade 7 {subject} Examination
        - Candidate Instructions
        - Section A, B, etc. with headings
        - Each question has marks
        - Provide sample answers after the questions
        - Use Zimbabwean exam formatting and style
        """
    else:
        template = f"""
        You are a revision paper generator for Grade 7 students in Zimbabwe studying {subject}.

        Use the context below to ensure curriculum relevance:
        ```{{context}}```

        Prompt: {{question}}

        You must generate a mock revision exam paper:
        - Title: Grade 7 {subject} Revision Paper
        - Short candidate instructions
        - Full set of exam-style questions organized into sections
        - Exclude answer key to encourage practice (no answers provided)
        - Keep language simple and clear
        - Indicate marks per question
        """

    prompt_template = PromptTemplate(template=template, input_variables=["context", "question"])

    retriever = vectorstore.as_retriever()
    docs = retriever.get_relevant_documents(prompt)
    context = "\n\n".join(doc.page_content for doc in docs)

    llm = ChatOpenAI(model_name=MODEL_NAME)
    chain = (
        {"context": lambda _: context, "question": RunnablePassthrough()} |
        prompt_template |
        llm |
        StrOutputParser()
    )

    result = chain.invoke(prompt)
    return result

# === Streamlit App ===
st.set_page_config(page_title="Exam Generatoration Chatbot", layout="centered")
st.title("📘 Exam Generation Bot for Zimsec Grade 7 subjects")

st.write("OpenAI API key loaded:", bool(os.getenv("OPENAI_API_KEY")))

st.markdown("This tool helps Teachers and Students generate mock and formal exams based on Zimbabwe's heritage based curriculum.")

role = st.selectbox("Select your role", ["Select"] + VALID_ROLES)
subject = st.selectbox("Select Subject", ["Select"] + VALID_SUBJECTS)

if subject != "Select" and role != "Select":
    prompt = st.text_area("Enter your prompt", placeholder="e.g. Create an end of term exam on algebra")
    if st.button("Generate Exam") and prompt:
        with st.spinner("Generating exam paper..."):
            try:
                output = generate_exam_response(role, subject, prompt)
                st.subheader("📄 Generated Exam Paper")
                st.code(output)

                # Convert to BytesIO for download
                buffer = BytesIO()
                buffer.write(output.encode("utf-8"))
                buffer.seek(0)
                file_name = f"{role}_{subject.replace(' ', '_')}_exam.txt"

                st.download_button(
                    label="⬇️ Download Exam Paper",
                    data=buffer,
                    file_name=file_name,
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.info("Please select your role and subject.")
