import argparse

import sentencepiece as spm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', default='TODO', type=str)
    parser.add_argument('--domain_sp_model_name', default='TODO', type=str)
    parser.add_argument('--max_sentence_length', default=16384, type=int)
    parser.add_argument('--pad_id', default=3, type=int)
    parser.add_argument('--vocab_size', default=2236, type=int)
    parser.add_argument('--model_type', default="BPE", type=str)

    args = parser.parse_args()
    print(args)

    spm.SentencePieceTrainer.train(
        input=args.in_file,
        model_prefix=args.domain_sp_model_name,
        shuffle_input_sentence=False,
        train_extremely_large_corpus=True,
        max_sentence_length=args.max_sentence_length,
        pad_id=args.pad_id,
        model_type=args.model_type,
        vocab_size=args.vocab_size,
        split_digits=True,
        split_by_unicode_script=True,
        byte_fallback=True,
        allow_whitespace_only_pieces=True,
        remove_extra_whitespaces=False,
        normalization_rule_name="nfkc",
    )

    # makes segmenter instance and loads the model file (m.model)
    sp = spm.SentencePieceProcessor()
    model_file = args.domain_sp_model_name + '.model'
    sp.load(model_file)

    # encode: text => id
    print(sp.encode_as_pieces('VietAI là tổ chức phi lợi nhận,this is a test'))
    print(sp.encode_as_ids('this is a test'))

    # decode: id => text
    print(sp.decode_pieces(['▁This', '▁is', '▁a', '▁t', 'est']))
    # print(sp.decode_ids([209, 31, 9, 375, 586]))


if __name__ == '__main__':
    main()