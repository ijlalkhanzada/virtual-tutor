import json

# Load dataset.json
try:
    with open("dataset.json", "r", encoding="utf-8") as file:
        dataset = json.load(file)
except FileNotFoundError:
    print("Error: 'dataset.json' file not found.")
    exit()

# Fix answer_start to be a list if it is an integer
for entry in dataset:
    if isinstance(entry["answers"]["answer_start"], int):
        entry["answers"]["answer_start"] = [entry["answers"]["answer_start"]]

# Save the fixed dataset to a new file
with open("dataset_fixed.json", "w", encoding="utf-8") as file:
    json.dump(dataset, file, ensure_ascii=False, indent=4)

print("Fixed dataset saved as 'dataset_fixed.json'")
