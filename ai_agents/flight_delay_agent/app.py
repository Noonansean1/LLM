import streamlit as st
from agent import get_agent

st.set_page_config(page_title="Flight Delay Checker", page_icon="✈️")
st.title("✈️ Flight Delay AI Agent")

user_input = st.text_input("Enter a question with flight number and date:")

if user_input:
    with st.spinner("Checking flight delay..."):
        agent = get_agent()
        response = agent.run(user_input)
    st.success(response)
