import sys
from pathlib import Path
import nltk


def get_dataset_file_paths(src_dir):
    all_paths = []
    for file_path in Path(src_dir).glob('*.txt'):
        all_paths.append(file_path)

    all_paths.sort()

    train_file_paths = [fp for i, fp in enumerate(all_paths) if i % 10 != 0]
    validate_file_paths = [fp for i, fp in enumerate(all_paths) if i % 10 == 0]

    return train_file_paths, validate_file_paths


def get_pairs(file_paths):
    pairs = []
    for file_path in file_paths:
        prev_tokens = None
        with open(file_path) as f:
            for line in f:
                line = line[:-1]
                tokens = nltk.word_tokenize(line, language='english')
                if prev_tokens:
                    pairs.append((prev_tokens, tokens))
                prev_tokens = tokens

    return pairs


def main():
    src_dir = sys.argv[1]
    dest_dir = sys.argv[2]

    train_file_paths, validate_file_paths = get_dataset_file_paths(src_dir)
    train_pairs = get_pairs(train_file_paths)
    validate_pairs = get_pairs(validate_file_paths)

    for prefix, pairs in [('train', train_pairs), ('valid', validate_pairs)]:
        for lang, id in [('fr', 0), ('en', 1)]:
            path = Path(dest_dir) / f'selfdialog.{prefix}.tok.{lang}'
            with open(path, mode='w') as f:
                for pair in pairs:
                    f.write(' '.join(pair[id]))
                    f.write('\n')


if __name__ == '__main__':
    main()
