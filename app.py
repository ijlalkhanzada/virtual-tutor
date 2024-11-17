from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Initialize the Hugging Face pipeline
qa_pipeline = pipeline("question-answering")

# Define the input schema
class Query(BaseModel):
    question: str
    context: str

@app.get("/")
def home():
    return {"message": "Welcome to the Virtual Tutor Chatbot"}

@app.post("/ask/")
def ask_question(query: Query):
    # Use the Hugging Face pipeline with the input
    result = qa_pipeline(question=query.question, context=query.context)
    return {"answer": result['answer']}
