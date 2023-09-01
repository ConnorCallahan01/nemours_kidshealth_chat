import utils
import streamlit as st
from streaming import StreamHandler

from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
import streamlit as st
import openai
from langchain.chat_models import ChatOpenAI 
from langchain.memory import ConversationBufferWindowMemory
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.agents import AgentType, initialize_agent
import requests
from langchain.agents import Tool
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.memory import ConversationBufferMemory
from langchain.schema.messages import SystemMessage
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.prompts import PromptTemplate
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.schema.messages import SystemMessage
from langchain.prompts import MessagesPlaceholder
from chains import VectorDBChain

st.title("💬 Nemours KidsHealth Chatbot")
st.subheader("Trained on the section: General Health --> Pains, Aches, & Injuries")
st.write("Chat with KidsHealth data to get more custom information regarding your child's health!")
st.write("Ask questions like: 'My child fell of their bike a week ago and is still complaining of pain, what should I do?', 'My child has a fever and is complaining of a headache, what should I do?', 'My child has a sprained ankle, what should I do?'")
class ContextChatbot:

    def __init__(self):
        utils.configure_openai_api_key()
        self.openai_model = '3ca59ab0-4e95-48b6-9fb6-3ea60a051057'
    
    @st.cache_resource
    def setup_chain(_self):
        loader_pinecone = PyPDFDirectoryLoader("./Aches_Pains_&_Injuries/")
        pages = loader_pinecone.load()
        model_name = 'text-embedding-ada-002'
        embeddings_pinecone = OpenAIEmbeddings(model=model_name, openai_api_key="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")
        pinecone.init(api_key='3ca59ab0-4e95-48b6-9fb6-3ea60a051057',environment='gcp-starter')

        index_name = "buzzindex"
        index = pinecone.Index(index_name)
        text_field = "text"

        llm = ChatOpenAI(
            temperature=0.8,
            model_name="ft:gpt-3.5-turbo-0613:personal::7tds9xcy",
            openai_api_key="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"
        )
        conversational_memory = ConversationBufferWindowMemory(
            memory_key='chat_history',
            k=5,
            return_messages=True
        )
        vdb = VectorDBChain(
            index_name="buzzindex",
            environment='gcp-starter',
            pinecone_api_key='3ca59ab0-4e95-48b6-9fb6-3ea60a051057'
        )

        vdb_tool = Tool(
            name=vdb.name,
            func=vdb.query,
            description="This tool allows you to get answers to the query from the documents."
        )
        system_message = """You are a medical assistant answering questions that parents have about their childrens' health.
                                                    Use the tools given to retrieve information from the medical database in order to answer the user's questions.
                                                    Answer the questions thoroughly and to the best of your knowledge. If you don't know the answer, respond with: 'I'm not too sure about that one, I suggest raising that question to a medical professional at your next doctors's visit'. 
                                                    Make sure to ask follow up questions on how you can either improve your response or ask other follow up questions to continue the conversation.
                                                    If you get the sense that you have answered the user's questions and the user doesn't have anymore questions, you can end it off with a nice message and hope that the user is able to get the help they need. 
                                                    Only answer questions that are relevant to the data you are trained on. If there isn't a high relevance to the data you are trained on, simply tell the user that you can't help them with their question as it is out of your scope."""
        
        agent = initialize_agent(
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            tools=[vdb_tool],
            llm=llm,
            verbose=True,
            agent_kwargs={"system_message": system_message},
            memory=conversational_memory,
            handle_parsing_errors=True,
            
        )
        return agent
    
    @utils.enable_chat_history
    def main(self):
        chain = self.setup_chain()
        user_query = st.chat_input(placeholder="Ask me anything!")
        if user_query:
            utils.display_msg(user_query, 'user')
            with st.spinner("Thinking..."):
                # st_cb = StreamHandler(st.empty())
                response = chain.run(user_query)
                st.session_state.messages.append({"role": "assistant", "content": response})
        # st.write(st.session_state.messages)
if __name__ == "__main__":
    obj = ContextChatbot()
    obj.main()