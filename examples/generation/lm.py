from typing import Dict

import numpy as np
import torch
import torch.optim as optim
from allennlp.common.file_utils import cached_path
from allennlp.common.tqdm import Tqdm
from allennlp.common.util import START_SYMBOL, END_SYMBOL
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.iterators import BucketIterator
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.trainer import Trainer
from overrides import overrides

EMBEDDING_SIZE = 128
HIDDEN_SIZE = 128
CUDA_DEVICE = -1

class LanguageModelingReader(DatasetReader):
    def __init__(self,
                 tokens_per_instance: int = None,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._tokens_per_instance = tokens_per_instance

        self._output_indexer: Dict[str, TokenIndexer] = None
        for name, indexer in self._token_indexers.items():
            if isinstance(indexer, SingleIdTokenIndexer):
                self._output_indexer = {name: indexer}
                break
        else:
            self._output_indexer = {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        with open(file_path, "r") as text_file:
            instance_strings = text_file.readlines()

        if self._tokens_per_instance is not None:
            all_text = " ".join([x.replace("\n", " ").strip() for x in instance_strings])
            tokenized_text = self._tokenizer.tokenize(all_text)
            num_tokens = self._tokens_per_instance + 1
            tokenized_strings = []
            for index in Tqdm.tqdm(range(0, len(tokenized_text) - num_tokens, num_tokens - 1)):
                tokenized_strings.append(tokenized_text[index:(index + num_tokens)])
        else:
            tokenized_strings = [self._tokenizer.tokenize(s) for s in instance_strings]

        for tokenized_string in tokenized_strings:
            tokenized_string.insert(0, Token(START_SYMBOL))
            tokenized_string.append(Token(END_SYMBOL))
            input_field = TextField(tokenized_string[:-1], self._token_indexers)
            output_field = TextField(tokenized_string[1:], self._output_indexer)
            yield Instance({'input_tokens': input_field,
                            'output_tokens': output_field})

    @overrides
    def text_to_instance(self, sentence: str) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_string = self._tokenizer.tokenize(sentence)
        input_field = TextField(tokenized_string[:-1], self._token_indexers)
        output_field = TextField(tokenized_string[1:], self._output_indexer)
        return Instance({'input_tokens': input_field, 'output_tokens': output_field})


class RNNLanguageModel(Model):
    def __init__(self, vocab: Vocabulary, cuda_device=-1) -> None:
        super().__init__(vocab)
        self.cuda_device = cuda_device

        token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                    embedding_dim=EMBEDDING_SIZE)
        if cuda_device > -1:
            token_embedding = token_embedding.to(cuda_device)
        self.embedder = BasicTextFieldEmbedder({"tokens": token_embedding})

        self.rnn = PytorchSeq2SeqWrapper(
            torch.nn.LSTM(EMBEDDING_SIZE, HIDDEN_SIZE, batch_first=True))

        self.hidden2out = torch.nn.Linear(in_features=self.rnn.get_output_dim(),
                                          out_features=vocab.get_vocab_size('tokens'))
        if cuda_device > -1:
            self.hidden2out = self.hidden2out.to(cuda_device)
            self.rnn = self.rnn.to(cuda_device)


    def forward(self, input_tokens, output_tokens):
        mask = get_text_field_mask(input_tokens)
        embeddings = self.embedder(input_tokens)
        rnn_hidden = self.rnn(embeddings, mask)
        out_logits = self.hidden2out(rnn_hidden)
        loss = sequence_cross_entropy_with_logits(out_logits, output_tokens['tokens'], mask)

        return {'loss': loss}

    def generate(self, max_len=20):
        words = []
        state = (torch.zeros(1, 1, HIDDEN_SIZE), torch.zeros(1, 1, HIDDEN_SIZE))
        if self.cuda_device > -1:
            state = (state[0].to(self.cuda_device), state[1].to(self.cuda_device))
        word_idx = self.vocab.get_token_index(START_SYMBOL, 'tokens')

        for i in range(max_len):
            tokens = torch.tensor([[word_idx]])
            if self.cuda_device > -1:
                tokens = tokens.to(self.cuda_device)
            embeddings = self.embedder({'tokens': tokens})
            output, state = self.rnn._module(embeddings, state)
            output = self.hidden2out(output)
            dist = torch.softmax(output[0, 0], dim=0)
            word_idx = np.random.choice(a=self.vocab.get_vocab_size('tokens'),
                                        p=dist.detach().numpy())

            if word_idx == self.vocab.get_token_index(END_SYMBOL, 'tokens'):
                break
            words.append(self.vocab.get_token_from_index(word_idx, 'tokens'))

        return words


def main():
    reader = LanguageModelingReader()
    train_dataset = reader.read('data/mt/sentences.eng.10k.txt')

    # for inst in train_dataset:
    #     print(inst)

    vocab = Vocabulary.from_instances(
        train_dataset, min_count={'tokens': 5})

    iterator = BucketIterator(
        batch_size=32, sorting_keys=[("input_tokens", "num_tokens")])

    iterator.index_with(vocab)

    model = RNNLanguageModel(vocab, cuda_device=CUDA_DEVICE)

    optimizer = optim.Adam(model.parameters())

    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      patience=10,
                      num_epochs=5,
                      cuda_device=CUDA_DEVICE)
    
    trainer.train()

    print(model.generate())
    print(model.generate())
    print(model.generate())
    print(model.generate())
    print(model.generate())

if __name__ == '__main__':
    main()
