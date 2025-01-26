# app.py
import streamlit as st
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from langchain.prompts import PromptTemplate

# Function to load and process the PDF
def load_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to initialize the RAG system
def initialize_rag_system(pdf_text):
    # Step 1: Split the Text into Chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Adjust chunk size as needed
        chunk_overlap=100,  # Adjust overlap as needed
        separators=["\n\n", "\n", " ", ""]  # Split by paragraphs, then sentences
    )
    texts = text_splitter.split_text(pdf_text)

    # Step 2: Generate Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Step 3: Create a FAISS Vector Store
    vectorstore = FAISS.from_texts(texts, embeddings)

    # Step 4: Load Falcon Model and Tokenizer
    model_name = "tiiuae/falcon-7b-instruct"  # Use Falcon-7B-Instruct model
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Configure 4-bit quantization using BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,  # Use 4-bit quantization
        bnb_4bit_use_double_quant=True,  # Use double quantization for better memory efficiency
        bnb_4bit_quant_type="nf4",  # Use 4-bit NormalFloat quantization
        bnb_4bit_compute_dtype="float16",  # Use float16 for computations
        llm_int8_enable_fp32_cpu_offload=True  # Enable CPU offload for 4-bit quantization
    )

    # Load the model with 4-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # Automatically map model to available devices (GPU/CPU)
        quantization_config=quantization_config,  # Pass the quantization config
        low_cpu_mem_usage=True  # Optimize CPU memory usage
    )

    # Step 5: Create a Pipeline for Falcon
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=300,  # Adjust for longer or shorter responses
        temperature=0.1,  # Control randomness (lower = more deterministic)
        top_p=0.9,  # Nucleus sampling (higher = more diverse)
    )

    # Wrap the pipeline in LangChain's HuggingFacePipeline
    llm = HuggingFacePipeline(pipeline=pipe)

    # Step 6: Define a Custom Prompt
    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="You are an expert in understanding and summarizing documents. Read the following context carefully and provide a detailed, accurate, and well-structured answer to the question. If the context does not provide enough information, say 'I don't know'.\n\nQuestion:\n{question}\n\nAnswer:",
    )

    # Step 7: Create the RetrievalQA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": custom_prompt}  # Use the custom prompt
    )

    return qa_chain

# Streamlit App
def main():
    st.title("RAG System with Falcon-7B and Streamlit")
    st.write("Upload a PDF and ask questions about its content.")

    # Step 1: Upload PDF
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file is not None:
        # Load and process the PDF
        pdf_text = load_pdf(uploaded_file)

        # Initialize the RAG system
        qa_chain = initialize_rag_system(pdf_text)

        # Step 2: Ask Questions
        st.write("### Ask a Question")
        question = st.text_input("Enter your question:")
        if question:
            # Query the RAG system
            response = qa_chain.invoke(question)
            
            # Display only the question and answer
            st.write("### Question:")
            st.write(question)
            st.write("### Answer:")
            st.write(response['result'])

# Run the Streamlit app
if __name__ == "__main__":
    main()