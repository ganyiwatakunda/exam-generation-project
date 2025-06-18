# exam-generation-project
This project shows how the RAG works together with the LLM to produce Exam Questions for primary school
teachers mainly focused on the grade 7 Zimsec curriculum, using generative AI and information retrieval.
It also explains how the results are generated for a user given an input prompt.
When the model receives an input prompt, the system first embeds the user input
prompt, then it searches for the most relevant documents in the knowledge base and then retrieves the
information, and creates a new prompt which will command the LLM to generate a response which is
relevant to the information that has been retrieved from the knowledge base , and the LLM generates
an output back to the user. These processes have been well explained in the following text and also at
the end of this section there is a diagram which summarizes the whole process into 5 main stages which
will be shown by the diagram at the end of the section which are text embedding of the user prompt,
vector search, information retrieval, Integration of RAG with LLM and lastly the generation of the
output.


## Input = prompt like ‘Grade 7 Mathematics paper 1 Exam for end of term’
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
## Output = A mathematics exam with instructions, mark allocations and questions which is similar to
other mathematics exam papers
