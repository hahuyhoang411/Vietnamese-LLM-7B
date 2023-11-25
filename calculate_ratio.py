from datasets import load_dataset
import re
from transformers import AutoTokenizer

# Define the function to count tokens
def count_tokens(batch):
    # This function will receive a batch of texts
    return {"token_count": [len(re.findall(r'\w+|[^\w\s]', text)) for text in batch["text"]]}

# Load the dataset
dataset = load_dataset("HoangHa/CulturaX001part", split="train")

# Apply the function on the dataset in parallel
dataset_with_token_counts = dataset.map(count_tokens,
                                        batched=True,
                                        num_proc=8,
                                        remove_columns=dataset.column_names)

# The dataset now has an additional field "token_count" with the token counts for each entry
print(dataset_with_token_counts)
print(dataset_with_token_counts["token_count"][0])
# Calculate the total number of tokens in the dataset
total_tokens = sum(dataset_with_token_counts['token_count'])

# Output the total number of tokens
print(f"Total number of tokens in the dataset: {total_tokens}")

# Load the tokenizer for a specific model
model_name = "/llm-vie/Vietnamese-LLM-7B/tokenizer"  # Replace with your desired model
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"])

tokenized_datasets = dataset.map(tokenize_function,
                                     batched=True,
                                     num_proc=16,
                                     remove_columns=dataset.column_names)

print(tokenized_datasets)

# Define the function to count input_ids
def count_input_ids(batch):
    # This function will receive a batch and return a batch of input_id counts
    return {"input_id_count": [len(input_ids) for input_ids in batch["input_ids"]]}

# Apply the function on the dataset in parallel
dataset_with_input_id_counts = tokenized_datasets.map(count_input_ids, batched=True, num_proc=8)

# Calculate the total number of input_ids in the dataset
total_input_ids = sum(dataset_with_input_id_counts['input_id_count'])

# Output the total number of input_ids
print(f"Total number of input_ids in the dataset: {total_input_ids}")

print(f"Compression ratio: {total_input_ids / total_tokens}")
