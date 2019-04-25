import csv
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.optim as optim
from allennlp.data import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers.token import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure
from allennlp.training.trainer import Trainer
from overrides import overrides


EMBEDDING_SIZE = 128
HIDDEN_SIZE = 128

class NERDatasetReader(DatasetReader):
    def __init__(self, token_indexers: Dict[str, TokenIndexer]=None, lazy=False):
        super().__init__(lazy=lazy)
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def text_to_instance(self, tokens: List[Token], labels: List[str]=None):
        fields = {}

        text_field = TextField(tokens, self.token_indexers)
        fields['tokens'] = text_field
        if labels:
            fields['labels'] = SequenceLabelField(labels, text_field)

        return Instance(fields)

    def _convert_sentence(self, rows: List[Tuple[str]]) -> Tuple[List[Token], List[str]]:
        """Given a list of rows, returns tokens and labels."""
        _, tokens, _, labels = zip(*rows)
        tokens = [Token(t) for t in tokens]

        # NOTE: the original dataset seems to confuse gpe with geo, and the distinction
        # seems arbitrary. Here we replace both with 'gpe'
        labels = [label.replace('geo', 'gpe') for label in labels]
        return tokens, labels

    @overrides
    def _read(self, file_path: str):

        sentence = []
        with open(file_path, mode='r', encoding='utf-8', errors='ignore') as csv_file:
            next(csv_file)
            reader = csv.reader(csv_file)

            for row in reader:
                if row[0] and sentence:
                    tokens, labels = self._convert_sentence(sentence)
                    yield self.text_to_instance(tokens, labels)

                    sentence = [row]
                else:
                    sentence.append(row)

            if sentence:
                tokens, labels = self._convert_sentence(sentence)
                yield self.text_to_instance(tokens, labels)


class LstmTagger(Model):
    def __init__(self,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        self.hidden2labels = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                             out_features=vocab.get_vocab_size('labels'))
        self.accuracy = CategoricalAccuracy()
        self.f1 = SpanBasedF1Measure(vocab, tag_namespace='labels')

    def forward(self,
                tokens: Dict[str, torch.Tensor],
                labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(tokens)
        embeddings = self.embedder(tokens)
        encoder_out = self.encoder(embeddings, mask)
        logits = self.hidden2labels(encoder_out)
        output = {'logits': logits}
        if labels is not None:
            self.accuracy(logits, labels, mask)
            self.f1(logits, labels, mask)
            output['loss'] = sequence_cross_entropy_with_logits(logits, labels, mask)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        f1_metrics = self.f1.get_metric(reset)
        return {'accuracy': self.accuracy.get_metric(reset),
                'prec': f1_metrics['precision-overall'],
                'rec': f1_metrics['recall-overall'],
                'f1': f1_metrics['f1-measure-overall']}


def predict(tokens: List[str], model: LstmTagger) -> List[str]:
    token_indexers = {'tokens': SingleIdTokenIndexer()}
    tokens = [Token(t) for t in tokens]
    inst = Instance({'tokens': TextField(tokens, token_indexers)})
    logits = model.forward_on_instance(inst)['logits']
    label_ids = np.argmax(logits, axis=1)
    labels = [model.vocab.get_token_from_index(label_id, 'labels')
              for label_id in label_ids]
    return labels


def main():
    reader = NERDatasetReader()
    dataset = list(reader.read('data/entity-annotated-corpus/ner_dataset.csv'))

    train_dataset = [inst for i, inst in enumerate(dataset) if i % 10 != 0]
    dev_dataset = [inst for i, inst in enumerate(dataset) if i % 10 == 0]

    vocab = Vocabulary.from_instances(train_dataset + dev_dataset)

    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                embedding_dim=EMBEDDING_SIZE)
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

    lstm = PytorchSeq2SeqWrapper(
        torch.nn.LSTM(EMBEDDING_SIZE, HIDDEN_SIZE, batch_first=True))

    model = LstmTagger(word_embeddings, lstm, vocab)

    optimizer = optim.Adam(model.parameters())

    iterator = BucketIterator(batch_size=16, sorting_keys=[("tokens", "num_tokens")])

    iterator.index_with(vocab)

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=dev_dataset,
                      patience=10,
                      num_epochs=10)
    trainer.train()

    tokens = ['Apple', 'is', 'looking', 'to', 'buy', 'U.K.', 'startup', 'for', '$1', 'billion', '.']
    labels = predict(tokens, model)
    print(' '.join('{}/{}'.format(token, label) for token, label in zip(tokens, labels)))


if __name__ == '__main__':
    main()
