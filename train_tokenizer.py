
from datasets import load_dataset
from tokenizers import SentencePieceBPETokenizer
from transformers import PreTrainedTokenizerFast

# Build a tokenizer
special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<cls>", "<sep>", "<mask>"]
tk_tokenizer = SentencePieceBPETokenizer()
# Initialize a dataset
dataset = load_dataset("HoangHa/CulturaX001part", num_proc=8, split="train")

# Build an iterator over this dataset
def batch_iterator(batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i: i + batch_size]["text"]

# And finally train
tk_tokenizer.train_from_iterator(
    batch_iterator(),
    vocab_size=16000,
    min_frequency=2,
    show_progress=True,
    special_tokens=special_tokens
)

tk_tokenizer.save("./vie-sbpe/tokenizer.json")

# convert
tokenizer = PreTrainedTokenizerFast(tokenizer_object=tk_tokenizer, model_max_length=2048, special_tokens=special_tokens)
tokenizer.bos_token = "<s>"
tokenizer.bos_token_id = tk_tokenizer.token_to_id("<s>")
tokenizer.pad_token = "<pad>"
tokenizer.pad_token_id = tk_tokenizer.token_to_id("<pad>")
tokenizer.eos_token = "</s>"
tokenizer.eos_token_id = tk_tokenizer.token_to_id("</s>")
tokenizer.unk_token = "<unk>"
tokenizer.unk_token_id = tk_tokenizer.token_to_id("<unk>")
tokenizer.cls_token = "<cls>"
tokenizer.cls_token_id = tk_tokenizer.token_to_id("<cls>")
tokenizer.sep_token = "<sep>"
tokenizer.sep_token_id = tk_tokenizer.token_to_id("<sep>")
tokenizer.mask_token = "<mask>"
tokenizer.mask_token_id = tk_tokenizer.token_to_id("<mask>")
# and save for later!
tokenizer.save_pretrained("./tokenizer")

from huggingface_hub import login
login()
tokenizer.push_to_hub("HoangHa/vietnamese-llm-7b")
