from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from PyPDF2 import PdfReader
import json
import threading

app = FastAPI()

# Lock for thread safety in dataset updates
dataset_lock = threading.Lock()

# Temporary storage for user contexts
user_contexts = {}

# Load context from JSON file (at startup)
def load_context_from_file(filename="contexts.json"):
    try:
        with open(filename, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

# Save context to JSON file
def save_context_to_file(context_data, filename="contexts.json"):
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(context_data, file, ensure_ascii=False, indent=4)

# Extract text from PDF
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Update dataset.json with new data
def update_dataset_safe(user_id, context, question="یہ سوال آپ کے context سے متعلق ہے۔"):
    with dataset_lock:
        try:
            # Load existing dataset
            with open("dataset.json", "r", encoding="utf-8") as file:
                dataset = json.load(file)
        except FileNotFoundError:
            dataset = []

        # Add new entry to dataset
        dataset.append({
            "context": context,
            "question": question,
            "answers": {
                "text": ["یہ جواب context میں موجود ہے۔"],
                "answer_start": [0]
            }
        })

        # Save updated dataset
        with open("dataset.json", "w", encoding="utf-8") as file:
            json.dump(dataset, file, ensure_ascii=False, indent=4)

# Root Endpoint
@app.get("/")
def root():
    return {"message": "Welcome to the Virtual Tutor API!"}

# Upload Context Endpoint
@app.post("/upload_context/")
async def upload_context(user_id: str, file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    # Extract text from uploaded file
    extracted_text = extract_text_from_pdf(file.file)
    user_contexts[user_id] = extracted_text  # Save context in memory

    # Save context to contexts.json
    save_context_to_file(user_contexts)

    # Automatically update dataset.json
    update_dataset_safe(user_id, extracted_text)

    return {"message": "Context uploaded and dataset updated successfully!"}

# Question-Answering Endpoint
class QARequest(BaseModel):
    question: str
    context: str

@app.post("/ask/")
async def ask_question(request: QARequest):
    try:
        # Load fine-tuned model
        qa_pipeline = pipeline("question-answering", model="./fine_tuned_model", tokenizer="bert-base-multilingual-cased")
        response = qa_pipeline(question=request.question, context=request.context)
        return {"answer": response["answer"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
