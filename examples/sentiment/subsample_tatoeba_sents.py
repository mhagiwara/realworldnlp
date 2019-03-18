import sys
from collections import Counter, defaultdict
import random


TOP10_LANGUAGES = {'eng', 'tur', 'ita', 'epo', 'deu', 'fra', 'por', 'spa', 'hun', 'ber'}


def main(dataset_type):
    random.seed(0)

    lang_count = Counter()
    lang_sents = defaultdict(list)

    for line in sys.stdin:
        sent_id, lang_id, sent = line.strip().split('\t')
        sent_id = int(sent_id)
        lang_count[lang_id] += 1

        if dataset_type == 'train':
            if sent_id % 10 in {0, 1}:
                continue
        elif dataset_type == 'dev':
            if sent_id % 10 != 1:
                continue
        elif dataset_type == 'test':
            if sent_id % 10 != 0:
                continue

        lang_sents[lang_id].append(sent)

    sents_per_lang = {'train': 10000, 'dev': 1000, 'test': 1000}[dataset_type]

    for lang_id in TOP10_LANGUAGES:
        sents_chosen = random.sample(lang_sents[lang_id], k=sents_per_lang)
        for sent in sents_chosen:
            print('{}\t{}'.format(lang_id, sent))


if __name__ == '__main__':
    main(sys.argv[1])
