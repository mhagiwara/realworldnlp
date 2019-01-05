from typing import Dict
import numpy as np

import torch
import torch.optim as optim

from allennlp.data.dataset_readers import UniversalDependenciesDatasetReader
from allennlp.data.iterators import BucketIterator
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.trainer import Trainer

from realworldnlp.predictors import UniversalPOSPredictor

EMBEDDING_SIZE = 128
HIDDEN_SIZE = 128

class LstmTagger(Model):
    def __init__(self,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        self.hidden2tag = torch.nn.Linear(in_features=encoder.get_output_dim(),
                                          out_features=vocab.get_vocab_size('pos'))
        self.accuracy = CategoricalAccuracy()

    def forward(self,
                words: Dict[str, torch.Tensor],
                pos_tags: torch.Tensor = None,
                **args) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(words)
        embeddings = self.embedder(words)
        encoder_out = self.encoder(embeddings, mask)
        tag_logits = self.hidden2tag(encoder_out)
        output = {"tag_logits": tag_logits}
        if pos_tags is not None:
            self.accuracy(tag_logits, pos_tags, mask)
            output["loss"] = sequence_cross_entropy_with_logits(tag_logits, pos_tags, mask)

        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}


def main():
    reader = UniversalDependenciesDatasetReader()
    train_dataset = reader.read('data/ud-treebanks-v2.3/UD_English-EWT/en_ewt-ud-train.conllu')
    dev_dataset = reader.read('data/ud-treebanks-v2.3/UD_English-EWT/en_ewt-ud-dev.conllu')

    vocab = Vocabulary.from_instances(train_dataset + dev_dataset)

    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                embedding_dim=EMBEDDING_SIZE)
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})

    lstm = PytorchSeq2SeqWrapper(
        torch.nn.LSTM(EMBEDDING_SIZE, HIDDEN_SIZE, batch_first=True))

    model = LstmTagger(word_embeddings, lstm, vocab)

    optimizer = optim.Adam(model.parameters())

    iterator = BucketIterator(batch_size=16, sorting_keys=[("words", "num_tokens")])

    iterator.index_with(vocab)

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=dev_dataset,
                      patience=10,
                      num_epochs=5)
    trainer.train()

    predictor = UniversalPOSPredictor(model, reader)
    logits = predictor.predict(['The', 'dog', 'ate', 'the', 'apple', '.'])['tag_logits']
    tag_ids = np.argmax(logits, axis=-1)

    print([vocab.get_token_from_index(tag_id, 'pos') for tag_id in tag_ids])


if __name__ == '__main__':
    main()
