import os
import streamlit as st

from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI


# ========================================================
# 🔐 API KEY  (Replace with yours)
# ========================================================
api_key = "api key"
os.environ["GOOGLE_API_KEY"] = api_key


# ========================================================
# Streamlit UI Setup
# ========================================================
st.set_page_config(page_title="🏭 Manufacturing RAG Assistant", layout="wide")

st.title("🏭 Manufacturing Machine Maintenance RAG Assistant")

tab1, tab2 = st.tabs(["📘 Project Overview", "🤖 RAG Assistant"])


# ========================================================
# 📘 TAB 1 — Project Overview
# ========================================================
with tab1:
    st.subheader("🎯 Problem Statement")

    st.write("""
Modern manufacturing floors rely on multiple machines—each having:

- Separate manuals  
- Troubleshooting guides  
- Maintenance logs  
- Error code sheets  
- Safety documentation  
- Warranty/service notes  

---

📌 When a machine issue occurs, technicians waste time manually searching documents, leading to:

| Challenge | Impact |
|----------|--------|
| Delayed troubleshooting | Increased downtime and production loss |
| Knowledge locked in PDFs/manuals | Hard to access in real-time |
| Experts required for every problem | Dependency risk |
| Inconsistent fixes | Safety and compliance issues |
| New employees struggle to understand errors | Training time increases |

---

### ❓ Why RAG (Retrieval-Augmented Generation)?

A normal LLM can hallucinate or give generic answers because it doesn’t *actually* know your machines or manuals.

RAG solves this by:

- 🔍 Retrieving relevant text chunks from documentation  
- 🤖 Sending them to an LLM  
- 📌 Generating factual context-aware answers  

---

### 🚀 Benefits

- Faster troubleshooting  
- Reduced downtime  
- Preserved expert knowledge  
- Consistent and safe repair steps  
- Faster training and onboarding  

---

➡️ Move to the **RAG Assistant tab** to ask questions from the knowledge base.
""")


# ========================================================
# 🤖 TAB 2 — RAG ASSISTANT
# ========================================================
with tab2:

    st.subheader("Ask a Machine Maintenance Question")

    # Load dataset
    file_path = "new_error_dataset.csv"

    if not os.path.exists(file_path):
        st.error(f"❌ CSV file not found: {file_path}")
        st.stop()

    loader = CSVLoader(file_path=file_path)
    docs = loader.load()

    # Text chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Vector DB
    vectorstore = Chroma.from_documents(splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

    # System prompt
    system_prompt = """
You are an AI assistant that provides helpful and natural responses based only on the information available in the provided CSV dataset.

Guidelines:
1. If the answer exists in the dataset, respond clearly and naturally using that information.
2. If the user asks about a machine error, include both the cause and recommended fix steps in the response.
3. Do not make up or assume information that is not present in the dataset.
4. If the dataset does not contain relevant information, respond with:
   "The document does not contain information related to this query."

"""

    # RAG function
    def rag_answer(query):
        docs = retriever.invoke(query)
        context = "\n\n".join([d.page_content for d in docs])

        final_prompt = f"""
CONTEXT:
{context}

QUESTION:
{query}

Answer ONLY using the provided context.
"""

        messages = [
            ("system", system_prompt),
            ("human", final_prompt)
        ]

        response = llm.invoke(messages)
        return response.content

    user_query = st.text_input("Enter your question about machine failure:")

    if st.button("Get Answer"):
        if user_query.strip():
            st.subheader("Answer:")
            with st.spinner("Processing..."):
                st.write(rag_answer(user_query))
        else:
            st.warning("⚠️ Please enter a question.")


