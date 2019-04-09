from typing import Dict

import torch
import torch.optim as optim
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers.character_tokenizer import CharacterTokenizer
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.training.trainer import Trainer
from overrides import overrides

from examples.sentiment.sst_classifier import LstmClassifier

EMBEDDING_DIM = 16
HIDDEN_DIM = 16

class TatoebaSentenceReader(DatasetReader):
    def __init__(self, token_indexers: Dict[str, TokenIndexer]=None, lazy=False):
        super().__init__(lazy=lazy)
        self.tokenizer = CharacterTokenizer()
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def text_to_instance(self, tokens, label=None):
        fields = {}

        fields['tokens'] = TextField(tokens, self.token_indexers)
        if label:
            fields['label'] = LabelField(label)

        return Instance(fields)

    @overrides
    def _read(self, file_path: str):
        with open(file_path, "r") as text_file:
            for line in text_file:
                lang_id, sent = line.rstrip().split('\t')

                tokens = self.tokenizer.tokenize(sent)

                yield self.text_to_instance(tokens, lang_id)


def main():
    reader = TatoebaSentenceReader()
    train_set = reader.read('data/mt/sentences.top10langs.train.tsv')
    dev_set = reader.read('data/mt/sentences.top10langs.dev.tsv')

    vocab = Vocabulary.from_instances(train_set,
                                      min_count={'tokens': 3})
    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                embedding_dim=EMBEDDING_DIM)
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    encoder = PytorchSeq2VecWrapper(
        torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))

    positive_label = vocab.get_token_index('eng', namespace='labels')
    model = LstmClassifier(word_embeddings, encoder, vocab, positive_label=positive_label)

    optimizer = optim.Adam(model.parameters())

    iterator = BucketIterator(batch_size=32, sorting_keys=[("tokens", "num_tokens")])

    iterator.index_with(vocab)

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_set,
                      validation_dataset=dev_set,
                      num_epochs=10)

    trainer.train()


if __name__ == '__main__':
    main()
