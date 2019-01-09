import onnx
import caffe2.python.onnx.backend as onnx_caffe2_backend
import torch
import codecs
import numpy as np


def read_vocab(filename):
    vocab = {}
    with codecs.open(filename, encoding='utf-8') as f:
        for vocab_id, line in enumerate(f):
            vocab_item = line.strip()
            vocab[vocab_item] = vocab_id
    return vocab


def main():
    token_table = read_vocab('examples/pos/vocab/tokens.txt')
    tokens = ['The', 'dog', 'ate', 'the', 'apple', '.']
    token_ids = [token_table[token] for token in tokens]
    model = onnx.load('examples/pos/model.onnx')

    prepared_backend = onnx_caffe2_backend.prepare(model)
    tokens = torch.tensor([token_ids], dtype=torch.long)
    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.long)
    inputs = {'inputs': tokens.data.numpy(), 'mask.1': mask.data.numpy()}

    logits = prepared_backend.run(inputs)[0]
    tag_ids = np.argmax(logits, axis=-1)
    print(tag_ids)

if __name__ == '__main__':
    main()
