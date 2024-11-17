from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from transformers import pipeline
from PyPDF2 import PdfReader
import json
from typing import Optional

app = FastAPI()

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

# Load existing contexts at startup
user_contexts = load_context_from_file()

# Extract text from PDF
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Update dataset.json with new data
def update_dataset(user_id, context, question="یہ سوال آپ کے context سے متعلق ہے۔"):
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

@app.post("/upload_context/")
async def upload_context(user_id: str, file: UploadFile = File(...)):
    # Extract text from uploaded file
    extracted_text = extract_text_from_pdf(file.file)
    user_contexts[user_id] = extracted_text  # Save context in memory

    # Save context to contexts.json
    save_context_to_file(user_contexts)

    # Automatically update dataset.json
    update_dataset(user_id, extracted_text)

    return {"message": "Context uploaded and dataset updated successfully!"}
