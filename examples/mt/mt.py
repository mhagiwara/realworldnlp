import torch
import torch.optim as optim
import itertools
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
from allennlp.modules.attention.linear_attention import LinearAttention
from allennlp.modules.attention.bilinear_attention import BilinearAttention
from allennlp.modules.attention.dot_product_attention import DotProductAttention
from realworldnlp.bleu import BLEU
from allennlp.nn import util
from allennlp.nn.activations import Activation


EN_EMBEDDING_DIM = 256
ZH_EMBEDDING_DIM = 256
HIDDEN_DIM = 256
CUDA_DEVICE = 0

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

    # attention = LinearAttention(HIDDEN_DIM, HIDDEN_DIM, activation=Activation.by_name('tanh')())
    # attention = BilinearAttention(HIDDEN_DIM, HIDDEN_DIM)
    attention = DotProductAttention()

    max_decoding_steps = 20   # TODO: make this variable
    model = SimpleSeq2Seq(vocab, source_embedder, encoder, max_decoding_steps,
                          target_embedding_dim=ZH_EMBEDDING_DIM,
                          target_namespace='target_tokens',
                          attention=attention,
                          beam_size=8)
    optimizer = optim.Adam(model.parameters())
    iterator = BucketIterator(batch_size=32, sorting_keys=[("source_tokens", "num_tokens")])

    iterator.index_with(vocab)

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      num_epochs=1,
                      cuda_device=CUDA_DEVICE)

    for _ in range(50):
        trainer.train()

        predictor = SimpleSeq2SeqPredictor(model, reader)
        bleu = BLEU()
        for batch in itertools.islice(iterator(validation_dataset), 10):
            batch = util.move_to_device(batch, CUDA_DEVICE)
            target_tokens = batch['target_tokens']['tokens']
            output_dict = model(**batch)
            predictions = output_dict['predictions']
            bleu(predictions, target_tokens)
        print(bleu.get_metric())

        for instance in itertools.islice(validation_dataset, 10):
            print('SOURCE:', instance.fields['source_tokens'].tokens)
            print('GOLD:', instance.fields['target_tokens'].tokens)
            print('PRED:', predictor.predict_instance(instance)['predicted_tokens'])


if __name__ == '__main__':
    main()
