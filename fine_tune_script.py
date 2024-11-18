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
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []

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

if not raw_data:
    print("Dataset is empty or invalid. Please check your dataset.json file.")
    exit()

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
        answer_text = examples["answers"][i]["text"][0]
        answer_start = examples["answers"][i]["answer_start"][0]

        start_positions.append(answer_start)
        end_positions.append(answer_start + len(answer_text))

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions

    return inputs

# Tokenize Dataset
model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Map preprocessing to dataset
try:
    tokenized_dataset = hf_dataset.map(preprocess_function, batched=True)
except Exception as e:
    print(f"Error during tokenization: {e}")
    exit()

# Split Dataset
if len(tokenized_dataset) < 2:
    print("Dataset is too small to split. Use the entire dataset for training.")
    train_dataset = tokenized_dataset
    eval_dataset = tokenized_dataset
else:
    split_dataset = tokenized_dataset.train_test_split(test_size=0.2)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
    report_to="none"  # Disable reporting for simplicity
)

# Data Collator
data_collator = DataCollatorWithPadding(tokenizer)

# Initialize Model
try:
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

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
try:
    trainer.train()
except Exception as e:
    print(f"Error during training: {e}")
    exit()

# Save Model
try:
    trainer.save_model("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")
    print("Fine-tuned model has been saved to './fine_tuned_model'.")
except Exception as e:
    print(f"Error saving the model: {e}")