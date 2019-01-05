from typing import List

from allennlp.common import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors import Predictor
from overrides import overrides


# You need to name your predictor and register so that `allennlp` command can recognize it
# Note that you need to use "@Predictor.register", not "@Model.register"!
@Predictor.register("sentence_classifier_predictor")
class SentenceClassifierPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)

    def predict(self, tokens: List[str]) -> JsonDict:
        return self.predict_json({"tokens" : tokens})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        tokens = json_dict["tokens"]
        return self._dataset_reader.text_to_instance(tokens)


@Predictor.register("universal_pos_predictor")
class UniversalPOSPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)

    def predict(self, words: List[str]) -> JsonDict:
        return self.predict_json({"words" : words})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        words = json_dict["words"]
        # This is a hack - the second argument to text_to_instance is a list of POS tags
        # that has the same length as words. We don't need it for prediction sd
        # just pass words.
        return self._dataset_reader.text_to_instance(words, words)
