from typing import Dict

import numpy as np
import torch
import torch.optim as optim
from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import \
    StanfordSentimentTreeBankDatasetReader
from allennlp.data.iterators import BucketIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.trainer import Trainer

from realworldnlp.predictors import SentenceClassifierPredictor

EMBEDDING_DIM = 128
HIDDEN_DIM = 128


class LstmClassifier(Model):
    def __init__(self,
                 word_embeddings: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=vocab.get_vocab_size('labels'))
        self.accuracy = CategoricalAccuracy()
        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self,
                tokens: Dict[str, torch.Tensor],
                label: torch.Tensor = None) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        embeddings = self.word_embeddings(tokens)
        encoder_out = self.encoder(embeddings, mask)
        logits = self.hidden2tag(encoder_out)
        output = {"logits": logits}
        if label is not None:
            self.accuracy(logits, label)
            output["loss"] = self.loss_function(logits, label)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}


def main():
    reader = StanfordSentimentTreeBankDatasetReader()

    train_dataset = reader.read('data/stanfordSentimentTreebank/trees/train.txt')
    dev_dataset = reader.read('data/stanfordSentimentTreebank/trees/dev.txt')

    vocab = Vocabulary.from_instances(train_dataset + dev_dataset,
                                      min_count={'tokens': 3})
    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                embedding_dim=EMBEDDING_DIM)
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

    lstm = PytorchSeq2VecWrapper(
        torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))

    model = LstmClassifier(word_embeddings, lstm, vocab)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    iterator = BucketIterator(batch_size=32, sorting_keys=[("tokens", "num_tokens")])

    iterator.index_with(vocab)

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=dev_dataset,
                      patience=10,
                      num_epochs=10)

    trainer.train()

    predictor = SentenceClassifierPredictor(model, dataset_reader=reader)
    logits = predictor.predict(['This', 'is', 'the', 'best', 'movie', 'ever', '!'])['logits']
    label_id = np.argmax(logits)

    print(model.vocab.get_token_from_index(label_id, 'labels'))

if __name__ == '__main__':
    main()
