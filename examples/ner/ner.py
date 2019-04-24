from typing import Dict, List, Tuple

from allennlp.data import DatasetReader
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from overrides import overrides
from allennlp.data.instance import Instance
from allennlp.data.tokenizers.token import Token

import csv

class NERDatasetReader(DatasetReader):
    def __init__(self, token_indexers: Dict[str, TokenIndexer]=None, lazy=False):
        super().__init__(lazy=lazy)
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def text_to_instance(self, tokens, labels=None):
        fields = {}

        tokens = [Token(t) for t in tokens]
        text_field = TextField(tokens, self.token_indexers)
        fields['tokens'] = text_field
        if labels:
            fields['labels'] = SequenceLabelField(labels, text_field)

        return Instance(fields)

    def _convert_sentence(self, rows: List[Tuple[str]]):
        """Given a list of rows, returns tokens and labels."""
        _, tokens, _, labels = zip(*rows)
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


def main():
    reader = NERDatasetReader()
    dataset = reader.read('data/entity-annotated-corpus/ner_dataset.csv')

    for inst in dataset:
        print(inst)

if __name__ == '__main__':
    main()
