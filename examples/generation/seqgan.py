import re
from typing import Dict, List, Tuple, Set

import numpy as np
import torch
import torch.optim as optim
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.iterators import BasicIterator
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers.character_tokenizer import CharacterTokenizer
from allennlp.data.tokenizers.token import Token
from allennlp.data.vocabulary import Vocabulary, DEFAULT_PADDING_TOKEN
from allennlp.models import Model
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.seq2vec_encoders import CnnEncoder
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.trainer import Trainer

EMBEDDING_SIZE = 32
HIDDEN_SIZE = 256
BATCH_SIZE = 256
D_STEPS = 3
D_EPOCHS = 3
G_EPOCHS = 10

class Generator(Model):
    def __init__(self,
                 embedder: TextFieldEmbedder,
                 vocab: Vocabulary,
                 max_len: int = 60) -> None:
        super().__init__(vocab)

        self.embedder = embedder

        self.rnn = PytorchSeq2SeqWrapper(
            torch.nn.LSTM(EMBEDDING_SIZE, HIDDEN_SIZE, batch_first=True))

        self.hidden2out = torch.nn.Linear(in_features=self.rnn.get_output_dim(),
                                          out_features=vocab.get_vocab_size('tokens'))
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
        vocab_size = self.vocab.get_vocab_size('tokens')
        uniform = torch.ones(vocab_size) / vocab_size

        log_likelihood = 0.
        words = []
        state = (torch.zeros(1, 1, HIDDEN_SIZE), torch.zeros(1, 1, HIDDEN_SIZE))

        word_idx = start_symbol_idx

        for i in range(self.max_len):
            tokens = torch.tensor([[word_idx]])

            embeddings = self.embedder({'tokens': tokens})
            output, state = self.rnn._module(embeddings, state)
            output = self.hidden2out(output)

            dist = torch.softmax(output[0, 0], dim=0)
            # dist = .99 * dist + .01 * uniform

            word_idx = start_symbol_idx

            while word_idx in {start_symbol_idx, padding_symbol_idx}:
                word_idx = torch.multinomial(
                    dist, num_samples=1, replacement=False).item()

            log_likelihood += torch.log(dist[word_idx])

            if word_idx == end_symbol_idx:
                break

            token = Token(text=self.vocab.get_token_from_index(word_idx, 'tokens'))
            words.append(token)

        return words, log_likelihood


class Discriminator(Model):
    def __init__(self,
                 embedder: TextFieldEmbedder,
                 vocab: Vocabulary) -> None:
        super().__init__(vocab)
        self.embedder = embedder

        self.encoder = CnnEncoder(EMBEDDING_SIZE, num_filters=8)

        self.linear = torch.nn.Linear(in_features=self.encoder.get_output_dim(),
                                      out_features=vocab.get_vocab_size('labels'))

        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self,
                tokens: Dict[str, torch.Tensor],
                label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(tokens)

        embeddings = self.embedder(tokens)
        encoder_out = self.encoder(embeddings, mask)
        logits = self.linear(encoder_out)

        output = {"logits": logits}
        output["loss"] = self.loss_function(logits, label)

        return output


def read_shakespeare(all_chars: Set[str]=None) -> List[List[Token]]:
    tokenizer = CharacterTokenizer()
    sentences = []
    with open('data/shakespeare/hamlet.txt') as f:
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


def text_to_disc_instance(tokens: List[Token],
                          label: str,
                          token_indexers: Dict[str, TokenIndexer]):
    fields = {'tokens': TextField(tokens, token_indexers),
              'label': LabelField(label)}

    return Instance(fields)


def text_to_ml_instance(tokens: List[Token],
                        token_indexers: Dict[str, TokenIndexer]):
    tokens = list(tokens)   # shallow copy
    tokens.insert(0, Token(START_SYMBOL))
    tokens.append(Token(END_SYMBOL))

    input_field = TextField(tokens[:-1], token_indexers)
    output_field = TextField(tokens[1:], token_indexers)
    return Instance({'input_tokens': input_field,
                     'output_tokens': output_field})


def get_discriminator_batch(generator: Generator,
                            train_set: List[List[Token]],
                            token_indexers: Dict[str, TokenIndexer]) -> List[Instance]:
    # Generate real batch
    instances = []
    sent_ids = np.random.choice(len(train_set), size=BATCH_SIZE, replace=False)
    for sent_id in sent_ids:
        tokens = train_set[sent_id]
        instance = text_to_disc_instance(tokens, 'real', token_indexers)
        instances.append(instance)

    # Generate fake batch
    num_fake_instances = 0
    while num_fake_instances < BATCH_SIZE:
        words, _ = generator.generate()
        if not words:
            continue

        instance = text_to_disc_instance(words, 'fake', token_indexers)
        instances.append(instance)
        num_fake_instances += 1

    assert len(instances) == 2 * BATCH_SIZE
    return instances


def get_generator_batch(generator: Generator,
                        token_indexers: Dict[str, TokenIndexer]) -> List[Tuple[Instance, torch.tensor]]:
    instances = []

    num_instances = 0
    while num_instances < 2 * BATCH_SIZE:
        words, log_likelihood = generator.generate()
        if not words:
            continue
        # HACK: add paddings
        while len(words) <= 4:
            words.append(Token(text=DEFAULT_PADDING_TOKEN))

        instance = text_to_disc_instance(words, 'fake', token_indexers)
        instances.append((instance, log_likelihood))

        num_instances += 1

    return instances

def get_reward(instance: Instance,
               discriminator: Discriminator,
               vocab: Vocabulary) -> float:

    logits = discriminator.forward_on_instance(instance)['logits']
    probs = np.exp(logits) / sum(np.exp(logits))  # softmax

    real_label_id = vocab.get_token_index('real', 'labels')
    return probs[real_label_id]


def main():
    generator_lr = 1.e-3
    print('generator_lr: {}'.format(generator_lr))

    all_chars = {END_SYMBOL, START_SYMBOL}
    all_chars.update("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,!?'-")
    token_counts = {char: 1 for char in all_chars}
    label_counts = {'real': 1, 'fake': 1}

    vocab = Vocabulary({'tokens': token_counts, 'labels': label_counts})
    token_indexers = {'tokens': SingleIdTokenIndexer()}

    train_set = read_shakespeare(all_chars=all_chars)

    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                embedding_dim=EMBEDDING_SIZE)
    embedder = BasicTextFieldEmbedder({"tokens": token_embedding})

    generator = Generator(embedder, vocab)
    discriminator = Discriminator(embedder, vocab)

    generator_optim = optim.Adam(generator.parameters(), lr=generator_lr)
    discriminator_optim = optim.Adagrad(discriminator.parameters())

    # pre-train generator
    instances = [text_to_ml_instance(tokens, token_indexers)
                 for tokens in train_set]
    iterator = BasicIterator(batch_size=BATCH_SIZE)
    iterator.index_with(vocab)

    trainer = Trainer(model=generator,
                      optimizer=generator_optim,
                      iterator=iterator,
                      train_dataset=instances,
                      num_epochs=G_EPOCHS)
    trainer.train()

    for epoch in range(500):
        # train discriminator
        fake_logs = []
        real_logs = []
        for d_step in range(D_STEPS):
            instances = get_discriminator_batch(generator, train_set, token_indexers)
            if d_step == 0:
                for inst in instances:
                    text = ''.join(token.text for token in inst.fields['tokens'])
                    label = inst.fields['label'].label
                    if label == 'real' and len(real_logs) < 10:
                        real_logs.append('    {:70s} {}'.format(text, label))
                    if label == 'fake' and len(fake_logs) < 10:
                        fake_logs.append('    {:70s} {}'.format(text, label))

            iterator = BasicIterator(batch_size=2 * BATCH_SIZE)
            iterator.index_with(vocab)

            trainer = Trainer(model=discriminator,
                              optimizer=discriminator_optim,
                              iterator=iterator,
                              train_dataset=instances,
                              num_epochs=D_EPOCHS)
            trainer.train()

        print('epoch: {}, step-D'.format(epoch))
        for log in real_logs + fake_logs:
            print(log)

        # train generator
        generator.zero_grad()

        instances = get_generator_batch(generator, token_indexers)

        log_likelihoods = []
        rewards = []
        logs = []

        for instance, log_likelihood in instances:
            reward = get_reward(instance, discriminator, vocab)

            rewards.append(reward)
            log_likelihoods.append(log_likelihood)

            if len(logs) < 10:
                text = ''.join(token.text for token in instance.fields['tokens'])
                logs.append('    {:70s} {:4.3f}'.format(text, reward))


        baseline = sum(rewards) / len(instances)
        avr_loss = sum(-1. * (reward - baseline) * log_likelihood
                       for reward, log_likelihood in zip(rewards, log_likelihoods))
        avr_loss /= len(instances)

        avr_loss.backward()
        generator_optim.step()

        print('epoch: {}, loss: {}, avr_reward: {}'.format(epoch, avr_loss, baseline))
        for log in logs:
            print(log)

if __name__ == '__main__':
    main()
