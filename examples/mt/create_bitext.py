"""
Script to convert sentences.csv and links.csv (downloaded from Tatoeba.org)
to a TSV format containing bilingual, aligned sentences in target_language and src_language.
"""

import argparse
from collections import defaultdict


def read_sentences(filename, target_language, src_language, sentences_with_audio=None):
    """
    Read sentences.csv and returns a dict containing sentence information.

    Parameters:
        filename (str): filename of 'sentence.csv'
        target_language (str): target language
        src_language (str): src language
        sentences_with_audio (set of int): set of sentence ids with audio.
            If not None, limit the output to this set.
    Returns:
        dict from sentence id (int) to Sentence information, where
        sentence information is a dict with 'sent_id', 'lang', and 'text' keys.

        dict only contains sentences in target_language or src_language."""

    sentences = {}
    for line in open(filename):
        sent_id, lang, text = line.rstrip().split('\t')
        if lang == src_language or lang == target_language:
            sent_id = int(sent_id)
            if (sentences_with_audio is not None
                    and lang == target_language
                    and sent_id not in sentences_with_audio):
                continue
            sentences[sent_id] = {'sent_id': sent_id, 'lang': lang, 'text': text}
    return sentences


def read_sentences_with_audio(filename):
    """
    Read sentences_with_audio.cvs file and returns a set of all the sentence IDs with audio.

    Parameters:
        filename (str): filename of 'sentences_with_audio.csv' file.

    Returns:
        set of all the sentence ids (int).
    """
    sentences_with_audio = set()
    for line in open(filename):
        sent_id = int(line.rstrip())
        sentences_with_audio.add(sent_id)
    return sentences_with_audio


def read_links(filename):
    """
    Read links.csv and returns a dict containing links information.

    Args:
        filename (str): filename of 'links.csv'
    Returns:
        dict from sentence id (int) of a sentence and a set of all its translation sentence ids."""

    links = defaultdict(set)
    for line in open(filename):
        sent_id, trans_id = line.rstrip().split('\t')
        links[int(sent_id)].add(int(trans_id))
    return links


def generate_translation_pairs(sentences, links, target_language, src_language):
    """
    Given sentences and links, generate a list of sentence pairs in target and source languages.

    Parameters:
        sentences: dict of sentence information (returned by read_sentences())
        links: dict of links information (returned by read_links())
        target_language (str): target language
        src_language (str): src language
    Returns:
        list of sentence pairs (sentence info 1, sentence info 2)
        where sentence info 1 is in target_language and sentence info 2 in src_language.
    """
    translations = []
    for sent_id, trans_ids in links.items():
        # Links in links.csv are reciprocal, meaning that if (id1, id2) is in the file,
        # (id2, id1) is also in the file. So we don't have to check both directions.
        if sent_id in sentences and sentences[sent_id]['lang'] == target_language:
            for trans_id in trans_ids:
                if trans_id in sentences and sentences[trans_id]['lang'] == src_language:
                    translations.append((sentences[sent_id], sentences[trans_id]))
    return translations


def write_tsv(translations):
    """
    Write translations as TSV to stdout.

    Parameters:
        translations (list): list of sentence pairs returned by generate_translation_pairs()
    """
    for sent1, sent2 in translations:
        sent1_text = '{sent_id}\t{lang}\t{text}'.format(**sent1)
        sent2_text = '{sent_id}\t{lang}\t{text}'.format(**sent2)
        print("%s\t%s" % (sent1_text, sent2_text))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('languages', help='pair of [target language]_[from language]')
    parser.add_argument('sentences', help='filename of sentences.csv')
    parser.add_argument('links', help='filename of links.csv')
    parser.add_argument('--audio', help='filename of sentences_with_audio.csv')
    args = parser.parse_args()
    target_language, src_language = args.languages.split('_')

    if args.audio:
        sentences_with_audio = read_sentences_with_audio(args.audio)
    else:
        sentences_with_audio = None

    sentences = read_sentences(args.sentences, target_language, src_language,
                               sentences_with_audio=sentences_with_audio)

    links = read_links(args.links)

    translations = generate_translation_pairs(sentences, links, target_language, src_language)

    write_tsv(translations)

if __name__ == '__main__':
    main()
