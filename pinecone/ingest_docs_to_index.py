import os
from dotenv import load_dotenv
from langchain.document_loaders import ReadTheDocsLoader
from langchain.embeddings import OpenAIEmbeddings, SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from PyPDF2 import PdfReader
import pinecone

# dotenv to load environment variables
load_dotenv()
# Initialize Pinecone or use your existing initialization code here
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))

def ingest_docs(folder_path):
    """
    Ingests all the PDF documents in the specified folder and adds them to the Pinecone index.

    Parameters:
        folder_path (str): The path to the folder containing the PDF documents.

    Returns:
        None
    """
    # Get a list of all files in the specified folder
    pdf_files = [file for file in os.listdir(folder_path) if file.endswith('.pdf')]
    # Iterate through each PDF file
    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file) # get the path to the PDF file
        text = ""
        # Open the PDF file in binary mode
        with open(pdf_path, 'rb') as file:
            # Create a PDF reader object
            pdf_reader = PdfReader(file) # extract text from each page
            for page in pdf_reader.pages:
                    text += page.extract_text() # append all contents from each page to text

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, # change chunk size here
        chunk_overlap=50, # change chunk overlap here
        length_function=len
    )
    chunks = text_splitter.split_text(text=text) # split text into chunks


    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_KEY")) # change openai api key here
    print(f"Going to add {len(chunks)} chunks to Pinecone")
    Pinecone.from_texts(chunks, embeddings, index_name=os.getenv("PINECONE_INDEX_NAME"))
    print("****Loading to vectorestore done ***")

if __name__ == "__main__":
    ingest_docs("../files") # put your files path here

