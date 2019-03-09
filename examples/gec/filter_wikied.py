"""
Script to filter WikiEd corpus and remove edits that are NOT correcting errors.
We use a very simple heuristic:
if the LM score per token improves by a certain amount, the edit is valid.
"""

import sys
import kenlm
import spacy
nlp = spacy.load('en_core_web_sm')

def tokenize(text):
    doc = nlp(text)
    return [token.text for token in doc]


def is_valid_edit(err, cor, model):
    err_tokens, cor_tokens = tokenize(err), tokenize(cor)
    err_score = model.score(err, bos=True, eos=True)
    cor_score = model.score(cor, bos=True, eos=True)
    err_score_per_token = err_score / len(err_tokens)
    cor_score_per_token = cor_score / len(cor_tokens)

    # The sentence is not very common even after edit — invalid
    if cor_score_per_token < -4.:
        return False

    # LM improve due to the edit — valid
    if cor_score_per_token / err_score_per_token < .9:
        return True

    return False


def main():
    model = kenlm.Model('data/wikitext-2-raw/wiki.train.raw.arpa')

    for line in sys.stdin:
        err, cor = line.rstrip().split('\t')
        is_valid = is_valid_edit(err, cor, model)
        if is_valid:
            print('{}\t{}'.format(err, cor))


if __name__ == '__main__':
    main()
