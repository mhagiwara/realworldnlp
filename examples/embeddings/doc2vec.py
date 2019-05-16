from gensim.utils import simple_preprocess
from gensim.models.doc2vec import TaggedDocument, Doc2Vec

def read_corpus(file_path):
    with open(file_path) as f:
        for i, line in enumerate(f):
            yield TaggedDocument(simple_preprocess(line), [i])


def main():
    train_set = list(read_corpus('data/tatoeba/sentences.eng.200k.txt'))
    model = Doc2Vec(vector_size=256, min_count=3, epochs=30)
    model.build_vocab(train_set)
    model.train(train_set, total_examples=model.corpus_count, epochs=model.epochs)

    query_vec = model.infer_vector(['i', 'heard', 'a', 'dog', 'barking', 'in', 'the', 'distance'])
    sims = model.docvecs.most_similar([query_vec], topn=10)
    for doc_id, sim in sims:
        print('{:3.2f} {}'.format(sim, train_set[doc_id].words))


if __name__ == '__main__':
    main()
