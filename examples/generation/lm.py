import re
from typing import Dict, List, Tuple, Set

import torch
import torch.optim as optim
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.iterators import BasicIterator
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, CharacterTokenizer
from allennlp.data.vocabulary import Vocabulary, DEFAULT_PADDING_TOKEN
from allennlp.models import Model
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.trainer import Trainer

EMBEDDING_SIZE = 32
HIDDEN_SIZE = 256
BATCH_SIZE = 128

def read_dataset(all_chars: Set[str]=None) -> List[List[Token]]:
    tokenizer = CharacterTokenizer()
    sentences = []
    with open('data/mt/sentences.eng.10k.txt') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line = re.sub(' +', ' ', line)
            tokens = tokenizer.tokenize(line)
            if all_chars:
                tokens = [token for token in tokens if token.text in all_chars]
            sentences.append(tokens)

    return sentences

def tokens_to_lm_instance(tokens: List[Token],
                          token_indexers: Dict[str, TokenIndexer]):
    tokens = list(tokens)   # shallow copy
    tokens.insert(0, Token(START_SYMBOL))
    tokens.append(Token(END_SYMBOL))

    input_field = TextField(tokens[:-1], token_indexers)
    output_field = TextField(tokens[1:], token_indexers)
    return Instance({'input_tokens': input_field,
                     'output_tokens': output_field})

class RNNLanguageModel(Model):
    def __init__(self,
                 embedder: TextFieldEmbedder,
                 hidden_size: int,
                 max_len: int,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)

        self.embedder = embedder

        self.rnn = PytorchSeq2SeqWrapper(
            torch.nn.LSTM(EMBEDDING_SIZE, HIDDEN_SIZE, batch_first=True))

        self.hidden2out = torch.nn.Linear(in_features=self.rnn.get_output_dim(),
                                          out_features=vocab.get_vocab_size('tokens'))
        self.hidden_size = hidden_size
        self.max_len = max_len

    def forward(self, input_tokens, output_tokens):
        mask = get_text_field_mask(input_tokens)
        embeddings = self.embedder(input_tokens)
        rnn_hidden = self.rnn(embeddings, mask)
        out_logits = self.hidden2out(rnn_hidden)
        loss = sequence_cross_entropy_with_logits(out_logits, output_tokens['tokens'], mask)

        return {'loss': loss}

    def generate(self) -> Tuple[List[Token], torch.tensor]:

        start_symbol_idx = self.vocab.get_token_index(START_SYMBOL, 'tokens')
        end_symbol_idx = self.vocab.get_token_index(END_SYMBOL, 'tokens')
        padding_symbol_idx = self.vocab.get_token_index(DEFAULT_PADDING_TOKEN, 'tokens')

        log_likelihood = 0.
        words = []
        state = (torch.zeros(1, 1, self.hidden_size),
                 torch.zeros(1, 1, self.hidden_size))

        word_idx = start_symbol_idx

        for i in range(self.max_len):
            tokens = torch.tensor([[word_idx]])

            embeddings = self.embedder({'tokens': tokens})
            output, state = self.rnn._module(embeddings, state)
            output = self.hidden2out(output)

            log_prob = torch.log_softmax(output[0, 0], dim=0)

            dist = torch.exp(log_prob)

            word_idx = start_symbol_idx

            while word_idx in {start_symbol_idx, padding_symbol_idx}:
                word_idx = torch.multinomial(
                    dist, num_samples=1, replacement=False).item()

            log_likelihood += log_prob[word_idx]

            if word_idx == end_symbol_idx:
                break

            token = Token(text=self.vocab.get_token_from_index(word_idx, 'tokens'))
            words.append(token)

        return words, log_likelihood

def main():
    all_chars = {END_SYMBOL, START_SYMBOL}
    all_chars.update("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,!?'-")
    token_counts = {char: 1 for char in all_chars}
    vocab = Vocabulary({'tokens': token_counts})

    token_indexers = {'tokens': SingleIdTokenIndexer()}

    train_set = read_dataset(all_chars)
    instances = [tokens_to_lm_instance(tokens, token_indexers)
                 for tokens in train_set]

    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                embedding_dim=EMBEDDING_SIZE)
    embedder = BasicTextFieldEmbedder({"tokens": token_embedding})

    model = RNNLanguageModel(embedder=embedder,
                             hidden_size=HIDDEN_SIZE,
                             max_len=80,
                             vocab=vocab)

    iterator = BasicIterator(batch_size=BATCH_SIZE)
    iterator.index_with(vocab)

    optimizer = optim.Adam(model.parameters(), lr=5.e-3)

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=instances,
                      num_epochs=10)
    trainer.train()

    for _ in range(50):
        tokens, _ = model.generate()
        print(''.join(token.text for token in tokens))


if __name__ == '__main__':
    main()
