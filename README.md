# exam-generation-project

An AI-driven application designed to help **Grade 7 primary school teachers and students** in Zimbabwe generate subject-aligned exam papers and practice tests using **OpenAI**, **RAG (Retrieval-Augmented Generation)**, and a curated knowledge base of curriculum materials and past papers.

## 📌 Project Overview

This tool allows users to input prompts like _"Create a Grade 7 Mathematics End of Term Test"_ and automatically generates structured, relevant, and curriculum-aligned exams based on stored teaching resources.

**Target Users**:
🎓 **Teachers** – To generate official classroom exams.
📘 **Students** – To practice with sample test papers on-demand.

**Supported Subjects**:
- Mathematics
- English
- ICT
- Social Science
- 
## 🚀 Features
✅ Prompt-based exam/question generation
✅ Retrieval-Augmented Generation using LangChain + OpenAI
✅ Use of past papers and curriculum documents as knowledge base
✅ Transparent and explainable AI (view source for each question)
✅ Streamlit-based user-friendly interface
✅ Downloadable/printable exam output

---
## Input = prompt like ‘Create a Grade 7 Mathematics End of Term Test’
1. The information retrieval model transforms the users prompt into an embedding and then
searches the Knowledge base for similar embeddings
2. Information retrieval model queries knowledge base/ FAISS vector store for relevant data which
is the mathematics paper 1 exam papers
3. FAISS vector store uses K-means clustering algorithm to find the papers that are closest to
mathematics paper 1 exams vectors.
4. FAISS uses Product Quantitization which compresses the similar vectors into shorter codes
reducing memory usage and speeding up the search for similarity
5. FAISS uses Optimized Product Quantitization which rotates the data to better fit the
Quantitization grid which improves the accuracy of the compressed vector
6. The findings from the knowledge base are then added and then the RAG creates a new prompt
for the LLM component
7. New prompt = Original User prompt + the embedded context (the findings from the knowledge
base)
8. The generator creates an output based on the augmented prompt (new prompt)
9. The new prompt synthesizes the user prompt with the retrieved data and instructs the
generator to consider this data in its response
## Output = A mathematics exam with instructions, mark allocations and questions which is similar to other mathematics exam papers
