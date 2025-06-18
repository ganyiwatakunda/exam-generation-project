import streamlit as st
from io import BytesIO

# ---------------------
# System Overview
# ---------------------
st.title("📘 AI Exam & Revision Generator (Prototype)")

st.markdown("""
Welcome to the prototype of an **AI-powered exam generation system** tailored for Grade 7 students and teachers in Zimbabwe.
This system uses **RAG (Retrieval-Augmented Generation)** and **OpenAI** to produce relevant, curriculum-aligned exams or revision materials.
""")

with st.expander("🔍 How it works"):
    st.markdown("""
    1. **Select your role** – either as a **Teacher** or a **Student**.
    2. Choose a subject from the available Grade 7 curriculum subjects.
    3. Enter a prompt describing what you need (e.g., *"End of Term ICT Exam"*).
    4. The system (in a real version) would generate questions based on teaching materials and past papers.
    5. Teachers receive a downloadable **full exam paper**, while students get **revision practice questions**.
    """)

st.markdown("---")

# ---------------------
# Role & Subject Selection
# ---------------------
role = st.selectbox("Who are you?", ["Select", "Teacher", "Student"])
subject = st.selectbox("Select Subject", ["Select", "Mathematics", "English", "Social Science", "ICT"])

# Sample prompt guidance
if subject != "Select":
    st.markdown("💡 **Sample Prompts:**")
    if role == "Student":
        st.code(f"Give me 10 revision questions in {subject.lower()} on the topic of fractions.")
    elif role == "Teacher":
        st.code(f"Generate a full Grade 7 {subject.lower()} exam on environmental science with sections and marks.")

# ---------------------
# Prompt Input
# ---------------------
if subject != "Select":
    prompt = st.text_area(f"Enter your prompt for {subject} ({role})", placeholder="e.g. Create revision questions on algebra...")

    if st.button("Generate"):
        st.success("✅ Prompt accepted. (In full version, this would generate content.)")

        if role == "Teacher":
            st.markdown("### 📄 Sample Full Exam Paper (Placeholder)")
            st.text("Section A - Multiple Choice\n1. What is 2 + 2? (4 marks)\n...\n\nSection B - Structured Questions\n...")
            
            # Create dummy downloadable file
            fake_exam = BytesIO()
            fake_exam.write(b"Grade 7 Sample Exam Paper\nSubject: %s\nPrompt: %s\n\n[This would be generated dynamically.]" % (subject.encode(), prompt.encode()))
            fake_exam.seek(0)

            st.download_button(label="⬇️ Download Exam Paper",
                               data=fake_exam,
                               file_name=f"{subject}_Exam_Paper.txt",
                               mime="text/plain")
        elif role == "Student":
            st.markdown("### 📘 Sample Practice Questions (Placeholder)")
            st.markdown("- What is 5 + 3?\n- Define a noun.\n- What is the function of a CPU?\n- Describe the causes of the liberation struggle in Zimbabwe.")

else:
    st.info("👆 Please select a subject before entering a prompt.")

