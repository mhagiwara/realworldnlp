from typing import List

from allennlp.common import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors import Predictor
from overrides import overrides


class SentenceClassifierPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)

    def predict(self, tokens: List[str]) -> JsonDict:
        return self.predict_json({"tokens" : tokens})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        tokens = json_dict["tokens"]
        return self._dataset_reader.text_to_instance(tokens)

