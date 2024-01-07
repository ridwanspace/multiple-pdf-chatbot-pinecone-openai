import os
from typing import Any, Dict, List
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.vectorstores import Pinecone
import pinecone

# dotenv to load environment variables
load_dotenv()
# Initialize Pinecone or use your existing initialization code here
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))

def run_conversational_retrieval(query: str, chat_history: List[Dict[str, Any]] = []):
    # OpenAI embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_KEY"))
    # load Pinecone vector database as knowledge base
    docsearch = Pinecone.from_existing_index(
        embedding=embeddings,
        index_name=os.getenv("PINECONE_INDEX_NAME"),
    ) # change index name here
    
    # OpenAI chat model
    chat = ChatOpenAI(
        verbose=True,
        temperature=0,
        max_tokens=512,
        openai_api_key=os.getenv("OPENAI_KEY")
    )
    # create a conversational question-answering chain as retrieval qa
    qa = ConversationalRetrievalChain.from_llm(
        llm=chat,
        retriever=docsearch.as_retriever()
    )
    return qa({"question": query, "chat_history": chat_history})


