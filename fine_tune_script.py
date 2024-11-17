from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
import json

# Load custom dataset
def load_custom_dataset(filepath):
    """
    Load custom Q&A dataset from a JSON file.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# Prepare dataset
def prepare_data(dataset):
    """
    Convert dataset into Hugging Face format.
    """
    contexts = []
    questions = []
    answers = []

    for item in dataset:
        contexts.append(item["context"])
        questions.append(item["question"])
        answers.append({
            "text": item["answers"]["text"][0],
            "answer_start": item["answers"]["answer_start"][0]
        })

    return {"context": contexts, "question": questions, "answers": answers}

# Load Pre-trained Model and Tokenizer
model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Load dataset
dataset_path = "dataset_fixed.json"
raw_data = load_custom_dataset(dataset_path)
processed_data = prepare_data(raw_data)

# Tokenize the dataset
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    contexts = examples["context"]

    inputs = tokenizer(
        contexts, questions, truncation=True, padding="max_length", max_length=512
    )

    start_positions = []
    end_positions = []

    # Ensure "answers" key has correct structure
    for i in range(len(questions)):
        # Check if "answers" is a list or a single string
        if isinstance(examples["answers"]["text"], list):
            answer_text = examples["answers"]["text"][0]  # First element of list
        else:
            answer_text = examples["answers"]["text"]  # Single string

        if isinstance(examples["answers"]["answer_start"], list):
            answer_start = examples["answers"]["answer_start"][0]  # First element of list
        else:
            answer_start = examples["answers"]["answer_start"]  # Single integer

        start_positions.append(answer_start)
        end_positions.append(answer_start + len(answer_text))

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions

    return inputs


# Convert dataset to Hugging Face format
hf_dataset = Dataset.from_dict(processed_data)
tokenized_dataset = hf_dataset.map(preprocess_function, batched=True)

# Split Dataset into Train/Test
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)

# Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",  # Evaluate after each epoch
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",  # Save model after each epoch
    logging_dir="./logs",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],  # Use the training set
    eval_dataset=tokenized_dataset["test"],    # Use the validation set
    tokenizer=tokenizer,
    data_collator=None,  # Default data collator
)

# Train the Model
trainer.train()

# Save the Fine-Tuned Model
trainer.save_model("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
