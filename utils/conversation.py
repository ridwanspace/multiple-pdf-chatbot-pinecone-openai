import os
from typing import Any, Dict, List
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain, LLMChain
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
    # create a conversational question-answering chain as retrieval qa and add i don't know if this is needed
    pre_prompt = """[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\nGenerate the next agent response by answering the question. Answer it as succinctly as possible. You are provided several documents with titles. If the answer comes from different documents please mention all possibilities in your answer and use the titles to separate between topics or domains. If you cannot answer the question from the given documents, please state that you do not have an answer.\n"""
    prompt = pre_prompt + "CONTEXT:\n\n{context}\n" +"Question : {question}" + "[\INST]"
    openai_prompt = PromptTemplate(template=prompt, input_variables=["context", "question"])

    # create a conversational question-answering chain as retrieval qa
    qa = ConversationalRetrievalChain.from_llm(
        llm=chat,
        retriever=docsearch.as_retriever(),
        combine_docs_chain_kwargs={"prompt": openai_prompt}
       
    )

    
    
    return qa({"question": query, "chat_history": chat_history})


