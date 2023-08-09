from abacusai import PredictionClient
import streamlit as st

client = PredictionClient()


st.write()

st.title("💬 Nemours KidsHealth Chatbot")
st.subheader("Trained on the section: General Health --> Pains, Aches, & Injuries")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"is_user": False, "text": "How can I help you?"}]

for msg in st.session_state.messages:
    if str(msg["is_user"]).lower() == "true":
        st.chat_message(str(msg["is_user"]), avatar="🧑‍💻").write(msg["text"])
    else:
        st.chat_message(str(msg["is_user"]), avatar="🤖").write(msg["text"])
    

if prompt := st.chat_input():

    
    st.session_state.messages.append({"is_user": True, "text": prompt})
    st.chat_message(str(st.session_state.messages[-1]["is_user"]), avatar="🧑‍💻").write(prompt)
    with st.spinner("Thinking..."):
        output = client.get_chat_response(deployment_token='e920e2c9ff424c0ea036f1160af718a0', deployment_id='1592061aba', 
                                messages=st.session_state.messages, 
                                llm_name=None, 
                                num_completion_tokens=700, 
                                system_message="""You are a medical assistant answering questions that parents have about their childrens' health.
                                                    Answer the questions thoroughly and to the best of your knowledge. If you don't know the answer, respond with: 'I'm not too sure about that one, I suggest raising that question to a medical professional at your next doctors's visit'. 
                                                    Make sure to ask follow up questions on how you can either improve your response or ask other follow up questions to continue the conversation.
                                                    If you get the sense that you have answered the user's questions and the user doesn't have anymore questions, you can end it off with a nice message and hope that the user is able to get the help they need. 
                                                    Only answer questions that are relevant to the data you are trained on. If there isn't a high relevance to the data you are trained on, simply tell the user that you can't help them with their question as it is out of your scope.""", 
                                temperature=None, 
                                filter_key_values=None, 
                                chat_config=None)
    msg = output['messages']
    st.session_state.messages.append(msg[-1])
    
    with st.chat_message(str(msg[-1]["is_user"]), avatar="🤖"):
        st.write(msg[-1]["text"])
        st.caption("Sources from the documents that helped define this answer:")
        for i in output["search_results"][0]["results"][0:1]:
            st.caption(i["answer"])
            

