# Enhanced Backend for Exam Generator (RAG + OpenAI) with Download Support

import os
import dotenv
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

dotenv.load_dotenv(".env")  # Load your OpenAI API key from .env file

VALID_SUBJECTS = ["Mathematics", "English", "ICT", "Social Science"]
VALID_ROLES = ["Student", "Teacher"]

def generate_exam_response(role: str, subject: str, prompt: str) -> str:
    if not prompt or subject not in VALID_SUBJECTS or role not in VALID_ROLES:
        raise ValueError("Invalid role, subject, or prompt")

    # === Load or Build Vector Store ===
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    if os.path.exists(STORAGE_PATH):
        vectorstore = FAISS.load_local(STORAGE_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        loader = DirectoryLoader(RESOURCE_PATH)
        docs = loader.load()
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(STORAGE_PATH)

    # === Define Prompt Based on Role ===
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
    else:  # Student
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

    # === Search & Generate ===
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

    # === Save to File for Download ===
    filename = f"./output/{role}_{subject.replace(' ', '_')}_exam.txt"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as file:
        file.write(result)

    return result

# Example usage:
if __name__ == "__main__":
    print("Select role: Teacher or Student")
    role = input("Role: ").strip()
    print("Select subject: Mathematics, English, ICT, or Social Science")
    subject = input("Subject: ").strip()
    prompt = input("Enter your prompt: ").strip()

    try:
        output = generate_exam_response(role, subject, prompt)
        print("\n=== Generated Output ===\n")
        print(output)
        print("\n📄 Exam saved for download in the 'output' folder.")
    except Exception as e:
        print(f"Error: {e}")

