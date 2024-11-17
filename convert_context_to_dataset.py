import json

# Load existing contexts.json
with open("contexts.json", "r", encoding="utf-8") as file:
    contexts = json.load(file)

# Prepare dataset format
dataset = []
for user_id, context in contexts.items():
    dataset.append({
        "context": context,
        "question": "یہ سوال آپ کے context سے متعلق ہے۔",  # Add your custom question
        "answers": {
            "text": ["یہ جواب context میں موجود ہے۔"],  # Replace with real answers
            "answer_start": [0]  # Adjust answer_start based on your context
        }
    })

# Save to dataset.json
with open("dataset.json", "w", encoding="utf-8") as file:
    json.dump(dataset, file, ensure_ascii=False, indent=4)

print("Dataset has been converted and saved as dataset.json")
