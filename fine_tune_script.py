import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

# Load Dataset
def load_custom_dataset(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# Prepare Data
def prepare_data(dataset):
    contexts = []
    questions = []
    answers = []

    for item in dataset:
        contexts.append(item["context"])
        questions.append(item["question"])
        answers.append({
            "text": item["answers"]["text"],
            "answer_start": item["answers"]["answer_start"]
        })

    return {"context": contexts, "question": questions, "answers": answers}

# Load Dataset
dataset_path = "dataset.json"
raw_data = load_custom_dataset(dataset_path)
processed_data = prepare_data(raw_data)

# Convert to Hugging Face Dataset
hf_dataset = Dataset.from_dict(processed_data)

# Preprocess Function
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    contexts = examples["context"]

    inputs = tokenizer(
        contexts, questions, truncation=True, padding="max_length", max_length=512
    )

    start_positions = []
    end_positions = []

    for i in range(len(examples["answers"])):
        # Check if "answers" is a dictionary or a list
        answer_data = examples["answers"][i]

        # If "answers" is a dictionary
        if isinstance(answer_data, dict):
            answer_text = answer_data["text"]
            answer_start = answer_data["answer_start"]
        else:
            raise TypeError(f"Unexpected format for answers: {answer_data}")

        # If "text" and "answer_start" are lists, pick the first one
        if isinstance(answer_text, list):
            answer_text = answer_text[0]
        if isinstance(answer_start, list):
            answer_start = answer_start[0]

        start_positions.append(answer_start)
        end_positions.append(answer_start + len(answer_text))

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions

    return inputs

# Tokenize Dataset
model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenized_dataset = hf_dataset.map(preprocess_function, batched=True)

# Split Dataset
split_dataset = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]
# Dataset ka size check karein
if len(tokenized_dataset) > 1:
    # Agar dataset mein 1 se zyada entries hain to split karein
    split_dataset = tokenized_dataset.train_test_split(test_size=0.2)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
else:
    print("Dataset chhota hai. Pura dataset training ke liye use ho raha hai.")
    train_dataset = tokenized_dataset
    eval_dataset = tokenized_dataset  # Optional: Evaluation ke liye bhi pura dataset use ho sakta hai

# Trainer ka setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)
a
# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs"
)

# Data Collator
data_collator = DataCollatorWithPadding(tokenizer)

# Initialize Model
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train Model
trainer.train()

# Save Model
trainer.save_model("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

print("Fine-tuned model has been saved to './fine_tuned_model'.")
