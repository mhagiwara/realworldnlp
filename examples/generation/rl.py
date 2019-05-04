import re
from typing import List, Tuple

import numpy as np
import torch
import torch.optim as optim
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from nltk.translate.chrf_score import sentence_chrf

EMBEDDING_SIZE = 32
HIDDEN_SIZE = 256
BATCH_SIZE = 256


class RNNLanguageModel(Model):
    def __init__(self, vocab: Vocabulary) -> None:
        super().__init__(vocab)

        token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                    embedding_dim=EMBEDDING_SIZE)
        self.embedder = BasicTextFieldEmbedder({"tokens": token_embedding})

        self.rnn = PytorchSeq2SeqWrapper(
            torch.nn.LSTM(EMBEDDING_SIZE, HIDDEN_SIZE, batch_first=True))

        self.hidden2out = torch.nn.Linear(in_features=self.rnn.get_output_dim(),
                                          out_features=vocab.get_vocab_size('tokens'))

    def generate(self, max_len: int) -> Tuple[List[str], torch.Tensor]:

        start_symbol_idx = self.vocab.get_token_index(START_SYMBOL, 'tokens')
        end_symbol_idx = self.vocab.get_token_index(END_SYMBOL, 'tokens')

        log_likelihood = 0.
        words = []
        state = (torch.zeros(1, 1, HIDDEN_SIZE),
                 torch.zeros(1, 1, HIDDEN_SIZE))

        word_idx = start_symbol_idx

        for i in range(max_len):
            tokens = torch.tensor([[word_idx]])

            embeddings = self.embedder({'tokens': tokens})
            output, state = self.rnn._module(embeddings, state)
            output = self.hidden2out(output)

            log_prob = torch.log_softmax(output[0, 0], dim=0)
            dist = torch.exp(log_prob)

            word_idx = start_symbol_idx
            while word_idx == start_symbol_idx:
                word_idx = torch.multinomial(
                    dist, num_samples=1, replacement=False).item()

            log_likelihood += log_prob[word_idx]

            if word_idx == end_symbol_idx:
                break

            words.append(self.vocab.get_token_from_index(word_idx, 'tokens'))

        return words, log_likelihood


def read_shakespeare():
    lines = []
    with open('data/shakespeare/hamlet.txt') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line = re.sub(' +', ' ', line)
            lines.append(line)

    return lines


def calculate_reward(generated: str, train_set: List[str], num_lines=100) -> float:
    line_ids = np.random.choice(len(train_set), size=num_lines)

    chrf_total = 0.
    for line_id in line_ids:
        line = train_set[line_id]
        chrf = sentence_chrf(line, generated, min_len=2, max_len=6, beta=1.,
                             ignore_whitespace=False)

        chrf_total += chrf

    return chrf_total / num_lines


def main():
    train_set = read_shakespeare()
    all_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,!?\'')
    all_chars.update([END_SYMBOL, START_SYMBOL])
    token_counts = {char: 1 for char in all_chars}

    vocab = Vocabulary({'tokens': token_counts}, non_padded_namespaces=('tokens',))

    model = RNNLanguageModel(vocab)

    optimizer = optim.Adam(model.parameters())

    for epoch in range(500):
        model.zero_grad()

        num_instances = 0
        logs = []
        log_likelihoods = []
        rewards = []

        while num_instances < BATCH_SIZE:
            words, log_likelihood = model.generate(max_len=60)
            if not words:
                continue
            reward = calculate_reward(''.join(words), train_set)
            if len(logs) < 10:
                logs.append('    {:70s} {:4.3f}'.format(''.join(words), reward))

            log_likelihoods.append(log_likelihood)
            rewards.append(reward)

            num_instances += 1

        baseline = sum(rewards) / num_instances
        loss = sum(-1. * (reward - baseline) * log_likelihood
                   for reward, log_likelihood in zip(rewards, log_likelihoods))
        loss /= num_instances
        print('epoch: {}, loss: {}, avr_reward: {}'.format(epoch, loss, baseline))
        for log in logs:
            print(log)

        loss.backward()
        optimizer.step()

if __name__ == '__main__':
    main()
