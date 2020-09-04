from allennlp.data.tokenizers import PretrainedTransformerTokenizer

from allennlp_models.pair_classification.dataset_readers import SnliReader

BERT_MODEL = 'bert-base-cased'

def main():
    tokenizer = PretrainedTransformerTokenizer(model_name=BERT_MODEL, add_special_tokens=False)
    result = tokenizer.tokenize('The best movie ever!')
    print(result)
    reader = SnliReader(tokenizer=tokenizer)
    for instance in reader.read('https://realworldnlpbook.s3.amazonaws.com/data/snli/snli_1.0_dev.jsonl'):
        print(instance)


if __name__ == "__main__":
    main()
