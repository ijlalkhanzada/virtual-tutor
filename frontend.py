import streamlit as st
import requests

# Title
st.title("Virtual Tutor Chatbot")

# Description
st.write("Ask any question based on a given context and get instant answers.")

# Upload Context Section
st.header("Upload PDF Context")
user_id = st.text_input("Enter User ID:")
uploaded_file = st.file_uploader("Upload a PDF file for context", type=["pdf"])

if st.button("Upload Context"):
    if user_id and uploaded_file:
        try:
            # Make POST request to upload_context endpoint
            response = requests.post(
                "http://127.0.0.1:8000/upload_context/",
                files={"file": uploaded_file.getvalue()},
                data={"user_id": user_id}
            )
            if response.status_code == 200:
                st.success(response.json().get("message"))
            else:
                st.error(f"Error: {response.json().get('detail')}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please provide a User ID and upload a PDF file!")

# Question-Answer Section
st.header("Ask a Question")
question = st.text_input("Enter your question:")
context = st.text_area("Enter the context for the question:")

if st.button("Get Answer"):
    if question and context:
        try:
            # Make POST request to ask endpoint
            response = requests.post(
                "http://127.0.0.1:8000/ask/",
                json={"question": question, "context": context}
            )
            if response.status_code == 200:
                st.write("Answer:", response.json().get("answer"))
            else:
                st.error(f"Error: {response.json().get('detail')}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please provide both a question and context!")
