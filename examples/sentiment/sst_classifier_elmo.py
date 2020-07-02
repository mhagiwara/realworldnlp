import numpy as np
import torch
import torch.optim as optim
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.data.samplers import BucketBatchSampler
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.training.trainer import Trainer
from allennlp_models.classification.dataset_readers.stanford_sentiment_tree_bank import \
    StanfordSentimentTreeBankDatasetReader

from examples.sentiment.sst_classifier import LstmClassifier
from realworldnlp.predictors import SentenceClassifierPredictor

EMBEDDING_DIM = 128
HIDDEN_DIM = 128


def main():
    # In order to use ELMo, each word in a sentence needs to be indexed with
    # an array of character IDs.
    elmo_token_indexer = ELMoTokenCharactersIndexer()
    reader = StanfordSentimentTreeBankDatasetReader(
        token_indexers={'tokens': elmo_token_indexer})

    train_dataset = reader.read('data/stanfordSentimentTreebank/trees/train.txt')
    dev_dataset = reader.read('data/stanfordSentimentTreebank/trees/dev.txt')

    # Initialize the ELMo-based token embedder using a pre-trained file.
    # This takes a while if you run this script for the first time

    # Original
    # options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    # weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

    # Medium
    # options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_options.json"
    # weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_weights.hdf5"

    # Use the 'Small' pre-trained model
    options_file = ('https://s3-us-west-2.amazonaws.com/allennlp/models/elmo'
                    '/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json')
    weight_file = ('https://s3-us-west-2.amazonaws.com/allennlp/models/elmo'
                   '/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5')

    elmo_embedder = ElmoTokenEmbedder(options_file, weight_file)

    vocab = Vocabulary.from_instances(train_dataset + dev_dataset,
                                      min_count={'tokens': 3})

    # Pass in the ElmoTokenEmbedder instance instead
    embedder = BasicTextFieldEmbedder({"tokens": elmo_embedder})

    # The dimension of the ELMo embedding will be 2 x [size of LSTM hidden states]
    elmo_embedding_dim = 256
    lstm = PytorchSeq2VecWrapper(
        torch.nn.LSTM(elmo_embedding_dim, HIDDEN_DIM, batch_first=True))

    model = LstmClassifier(embedder, lstm, vocab)
    optimizer = optim.Adam(model.parameters())

    iterator = BucketIterator(batch_size=32, sorting_keys=[("tokens", "num_tokens")])

    iterator.index_with(vocab)

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=dev_dataset,
                      patience=10,
                      num_epochs=20)

    trainer.train()

    tokens = ['This', 'is', 'the', 'best', 'movie', 'ever', '!']
    predictor = SentenceClassifierPredictor(model, dataset_reader=reader)
    logits = predictor.predict(tokens)['logits']
    label_id = np.argmax(logits)

    print(model.vocab.get_token_from_index(label_id, 'labels'))

if __name__ == '__main__':
    main()
