import torch
import torch.optim as optim
from allennlp.data.dataset_readers.seq2seq import Seq2SeqDatasetReader
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers.character_tokenizer import CharacterTokenizer
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.encoder_decoders.simple_seq2seq import SimpleSeq2Seq
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.training.trainer import Trainer
from allennlp.predictors import SimpleSeq2SeqPredictor

EN_EMBEDDING_DIM = 128
ZH_EMBEDDING_DIM = 128
HIDDEN_DIM = 128


def main():
    reader = Seq2SeqDatasetReader(
        source_tokenizer=WordTokenizer(),
        target_tokenizer=CharacterTokenizer(),
        source_token_indexers={'tokens': SingleIdTokenIndexer()},
        target_token_indexers={'tokens': SingleIdTokenIndexer(namespace='target_tokens')})
    train_dataset = reader.read('data/mt/tatoeba.eng_cmn.train.tsv')
    validation_dataset = reader.read('data/mt/tatoeba.eng_cmn.dev.tsv')

    vocab = Vocabulary.from_instances(train_dataset + validation_dataset,
                                      min_count={'tokens': 3, 'target_tokens': 3})

    en_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                             embedding_dim=EN_EMBEDDING_DIM)
    encoder = PytorchSeq2SeqWrapper(
        torch.nn.LSTM(EN_EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))

    source_embedder = BasicTextFieldEmbedder({"tokens": en_embedding})
    max_decoding_steps = 20   # TODO: make this variable
    model = SimpleSeq2Seq(vocab, source_embedder, encoder, max_decoding_steps,
                          target_embedding_dim=ZH_EMBEDDING_DIM,
                          target_namespace='target_tokens')
    optimizer = optim.Adam(model.parameters())
    iterator = BucketIterator(batch_size=32, sorting_keys=[("source_tokens", "num_tokens")])

    iterator.index_with(vocab)

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=validation_dataset,
                      num_epochs=1)

    trainer.train()

    predictor = SimpleSeq2SeqPredictor(model, reader)
    print(predictor.predict('She loves Chinese food.')['predicted_tokens'])
    print(predictor.predict('My kids are at school.')['predicted_tokens'])


if __name__ == '__main__':
    main()
