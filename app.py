import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

#we are using text splitter to read the PDF and convert it into chunks
#Now for Converting data into vectors using embeddings

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv 

#FAISS is for vector embeddings
#helps us to do the chat and any kind of prompts we would need

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

#basically connected the genAi lib to the r equired model using the generated
#API key

#PDF reader will read the pdf imported
#inform of list
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text
#so basically we parse the pdf and grab all its text 

#Now dividing the data in chunks
def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks=text_splitter.split_text(text)
    return chunks

#Converting these chunks into vectors

def get_vector_store(text_chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store=FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

#here localstorage is the name of folder in  which stored vectors will be stored locally

def get_conversational_chain():

    prompt_template="""
    Answer the question as detailed as possible context, make sure to provide all the
    details, if the answer is not in the provided context just say, "answer is not available in the context",
    don't provide wrong answers.\n\n
    Context:\n{context}?\n
    Question:\n{question}\n
    Answer:
    """
    model=ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)

    prompt=PromptTemplate(template=prompt_template, input_variables=["context","question"])
    chain=load_qa_chain(model,chain_type="stuff",prompt=prompt)
    return chain
#here we first create a prompt template then load the model and then create chain
#that gives us the summarised output and finally return it

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db=FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)
    docs=new_db.similarity_search(user_question)
    chain=get_conversational_chain()
    response = chain(
        {"input_documents":docs, "question":user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ",response["output_text"])
     
#here we bascially recieve the use input and call the faiss index as it has 
#the pdf data and then we run similarity search inside the database(FAISS)
#then we call the get conversational chain to get the output
#finall we print it All realteed to the text box

#now making the frontend for this chatbot

def main():
    st.set_page_config("Chat with multiple PDF")
    st.header("Chat with Multiple PDF using Gemini 1.5 PRO")

    user_question=st.text_input("Ask a Question from the PDF files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs=st.file_uploader("Upload your PDF files and Click on te Submit Icon",accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing"):
                raw_text =  get_pdf_text(pdf_docs)
                text_chunks=get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done") 

if __name__=="__main__":
    main()

