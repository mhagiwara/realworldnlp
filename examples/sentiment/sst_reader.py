from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField, Field
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, SpacyTokenizer
from nltk.tree import Tree
from overrides import overrides
from typing import Dict, List


@DatasetReader.register("sst_with_tokenizer")
class StanfordSentimentTreeBankDatasetReaderWithTokenizer(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tokenizer: Tokenizer = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy=lazy)
        self._tokenizer = tokenizer or SpacyTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            for line in data_file.readlines():
                line = line.strip("\n")
                if not line:
                    continue
                parsed_line = Tree.fromstring(line)
                sent = ' '.join(parsed_line.leaves())
                tokens = self._tokenizer.tokenize(sent)
                label = parsed_line.label()
                instance = self.text_to_instance(tokens, label)
                if instance is not None:
                    yield instance

    @overrides
    def text_to_instance(self, tokens: List[Token], sentiment: str = None) -> Instance:
        text_field = TextField(tokens, token_indexers=self._token_indexers)
        fields: Dict[str, Field] = {"tokens": text_field}
        if sentiment is not None:
            fields['label'] = LabelField(sentiment)
        return Instance(fields)
