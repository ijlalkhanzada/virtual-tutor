import streamlit as st
import requests

st.title("Virtual Tutor Chatbot")

# User inputs
question = st.text_input("Enter your question:")
context = st.text_area("Enter context for the question:")

if st.button("Ask"):
    if question and context:
        response = requests.post("http://127.0.0.1:8000/ask/", json={"question": question, "context": context})
        st.write("Answer:", response.json().get("answer"))
    else:
        st.warning("Please provide both question and context!")
