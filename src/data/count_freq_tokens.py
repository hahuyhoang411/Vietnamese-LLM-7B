import json
import argparse

from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument("--token_name", type=str, default="", help="")
    parser.add_argument("--data_name", type=str, default="", help="")
    parser.add_argument("--data_config", type=str, default=None, help="")
    parser.add_argument("--column_text_name", type=str, default="text", help="")
    parser.add_argument("--max_length", type=int, default=512, help="")
    parser.add_argument("--num_proc", type=int, default=2, help="")
    parser.add_argument("--output_path", type=str, default="", help="")
    args = parser.parse_args()
    return args


def save_json(path, file_save):
    with open(path, "w") as f:
        return json.dump(file_save, f, ensure_ascii=False)


def ids_to_tokens(data, tokenizer, args):
    inputs = tokenizer.encode(
        data, max_length=args.max_length, padding=True, truncation=True
    )
    input_tokens = tokenizer.convert_ids_to_tokens(inputs)
    return input_tokens


def main():
    dict_count = {}
    args = parse_arguments()

    tokenizer = AutoTokenizer.from_pretrained(args.token_name, token=True)
    dataset = load_dataset(
        args.data_name, args.data_config, num_proc=args.num_proc, split="train"
    )
    dataset = dataset.map(
        lambda data: {
            "input_tokens": ids_to_tokens(data[args.column_text_name], tokenizer, args)
        }
    )

    for data in tqdm(dataset):
        for input_token in data["input_tokens"]:
            dict_count[input_token] = 1 + dict_count.get(input_token, 0)

    save_json(args.output_path, dict_count)


if __name__ == "__main__":
    main()
