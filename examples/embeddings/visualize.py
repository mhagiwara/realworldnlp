from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def read_glove(file_path):
    with open(file_path) as f:
        for i, line in enumerate(f):
            fields = line.rstrip().split(' ')
            vec = [float(x) for x in fields[1:]]
            word = fields[0]
            yield (word, vec)

            if i == 1000:
                break


def main():
    words = []
    vectors = []
    for word, vec in read_glove('data/glove/glove.42B.300d.txt'):
        words.append(word)
        vectors.append(vec)

    model = TSNE(n_components=2, init='pca', random_state=0)
    coordinates = model.fit_transform(vectors)

    plt.figure(figsize=(8, 8))

    for word, xy in zip(words, coordinates):
        plt.scatter(xy[0], xy[1])
        plt.annotate(word,
                     xy=(xy[0], xy[1]),
                     xytext=(2, 2),
                     textcoords='offset points')

    plt.xlim(25, 55)
    plt.ylim(-15, 15)
    plt.show()


if __name__ == '__main__':
    main()
