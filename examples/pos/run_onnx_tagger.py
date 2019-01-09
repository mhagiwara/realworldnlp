import onnx
import caffe2.python.onnx.backend as onnx_caffe2_backend
import torch
import codecs
import numpy as np


def read_vocab(filename):
    token2id = {}
    id2token = {}
    with codecs.open(filename, encoding='utf-8') as f:
        for token_id, line in enumerate(f):
            token = line.strip()
            # vocab_id is zero base because 0 is for padding
            token2id[token] = token_id + 1
            id2token[token_id + 1] = token
    return token2id, id2token


def main():
    token2id, id2token = read_vocab('examples/pos/vocab/tokens.txt')
    pos2id, id2pos = read_vocab('examples/pos/vocab/pos.txt')

    tokens = ['Time', 'flies', 'like', 'an', 'arrow', '.']
    token_ids = [token2id.get(token, token2id['@@UNKNOWN@@']) for token in tokens]
    model = onnx.load('examples/pos/model.onnx')

    prepared_backend = onnx_caffe2_backend.prepare(model)
    tokens = torch.tensor([token_ids], dtype=torch.long)
    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.long)
    inputs = {'inputs': tokens.data.numpy(), 'mask.1': mask.data.numpy()}

    logits = prepared_backend.run(inputs)[0]
    tag_ids = np.argmax(logits, axis=-1)[0]
    print([id2pos[tag_id] for tag_id in tag_ids])

if __name__ == '__main__':
    main()
