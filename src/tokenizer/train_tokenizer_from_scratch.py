import argparse
from tqdm import tqdm
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer


def parse_arguments():
    parser = argparse.ArgumentParser(description='...')
    parser.add_argument('--data_name', type=str, default='',
                        help='')
    parser.add_argument('--data_config', type=str, default=None,
                        help='')
    parser.add_argument('--batch_size', type=int, default='',
                        help='')
    parser.add_argument('--vocab_size', type=int, default='',
                        help='')
    parser.add_argument('--text_column_name', type=str, default='text',
                        help='')
    parser.add_argument('--output_path', type=str, default='',
                        help='')
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    dataset = load_dataset(args.data_name, args.data_config, split="train")
    tokenizer = ByteLevelBPETokenizer()

    def batch_iterator(batch_size=args.batch_size):
        for i in tqdm(range(0, len(dataset), batch_size)):
            yield dataset[i: i + batch_size][args.text_column_name]

    tokenizer.train_from_iterator(batch_iterator(), vocab_size=args.vocab_size, min_frequency=2, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])

    tokenizer.save(f"{args.output_path}/tokenizer.json")


if __name__ == "__main__":
    main()
