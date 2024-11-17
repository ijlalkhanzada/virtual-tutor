from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from PyPDF2 import PdfReader
import json
from typing import Optional
import torch

app = FastAPI()

# Temporary storage for user contexts
user_contexts = {}

# Load fine-tuned model and tokenizer
model_name = "./fine_tuned_model"  # Path to your fine-tuned model
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
except OSError as e:
    raise RuntimeError(f"Model loading failed. Make sure the fine-tuned model exists at {model_name}. Error: {str(e)}")

# Load context from JSON file
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

# Schema for user queries
class Query(BaseModel):
    question: str
    context: Optional[str] = None

@app.get("/")
def home():
    return {"message": "Welcome to the Virtual Tutor Chatbot"}

@app.post("/upload_context/")
async def upload_context(user_id: str, file: UploadFile = File(...)):
    """
    Upload a PDF file and store its content as context for the user.
    """
    extracted_text = extract_text_from_pdf(file.file)
    user_contexts[user_id] = extracted_text  # Save context in memory

    # Save context to contexts.json
    save_context_to_file(user_contexts)

    # Automatically update dataset.json
    update_dataset(user_id, extracted_text)

    return {"message": "Context uploaded and dataset updated successfully!"}

@app.post("/ask/")
def ask_question(user_id: str, query: Query):
    """
    Answer a question based on the provided or stored context.
    """
    # Check if the context is provided or stored
    if query.context is None:
        if user_id in user_contexts:
            query.context = user_contexts[user_id]
        else:
            raise HTTPException(status_code=400, detail="No context provided or stored for this user.")

    # Tokenize input
    inputs = tokenizer.encode_plus(
        query.question,
        query.context,
        add_special_tokens=True,
        return_tensors="pt"
    )

    # Get predictions from the model
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract start and end logits for the answer
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1

    # Convert token IDs to string
    input_ids = inputs["input_ids"].tolist()[0]
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
    )

    return {
        "question": query.question,
        "context": query.context,
        "answer": answer
    }
