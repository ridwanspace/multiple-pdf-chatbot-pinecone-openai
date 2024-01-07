import os
import streamlit as st
from dotenv import load_dotenv
from utils.conversation import run_conversational_retrieval
from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)

load_dotenv()

st.set_page_config(page_title = "Chat with Multiple PDFs Using Pinecone", page_icon = ":robot_face")

st.title("Multiple PDFs Chatbot with Streamlit, OpenAI, LangChain, and Pinecone :robot_face:")
st.markdown(
    """
    ### Technologies Used:
    1. [**OpenAI Embeddings**](https://python.langchain.com/docs/integrations/text_embedding/openai):

    Utilized for extracting contextual embeddings from the PDF content.
    
    2. [**LangChain**](https://python.langchain.com/docs/integrations/chat/openai):

    Employs OpenAI and Pinecone to process user queries and generate context-aware responses.
    
    3. [**Pinecone**](https://python.langchain.com/docs/integrations/vectorstores/pinecone):

    Used for indexing and efficient retrieval of embeddings generated from PDF content.

    4. [**Conversational Retrieval Chain**](https://medium.com/@jerome.o.diaz/langchain-conversational-retrieval-chain-how-does-it-work-bb2d71cbb665):

     A kind of chain used to be provided with a query and to answer it using documents retrieved from the query. It is one of the many possibilities to perform Retrieval-Augmented Generation.
    """
)




if (
    "chat_answers_history" not in st.session_state
    and "user_prompt_history" not in st.session_state
    and "chat_history" not in st.session_state
):
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []


prompt = st.text_input("Prompt", placeholder="Enter your question here...") or st.button(
    "Submit"
)

if prompt:
    with st.spinner("Generating response..."):
        generated_response = run_conversational_retrieval(
            query=prompt, chat_history=st.session_state["chat_history"]
        )

        
        formatted_response = (
            f"{generated_response['answer']}"
        )

        st.session_state.chat_history.append((prompt, generated_response["answer"]))
        st.session_state.user_prompt_history.append(prompt)
        st.session_state.chat_answers_history.append(formatted_response)

if st.session_state["chat_answers_history"]:
    for generated_response, user_query in zip(
        st.session_state["chat_answers_history"],
        st.session_state["user_prompt_history"],
    ):
        message(
            user_query,
            is_user=True,
            key=hash(user_query)
        )
        message(generated_response)