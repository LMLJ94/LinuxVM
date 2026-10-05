import numpy as np
from sklearn.cluster import MiniBatchKMeans

VOCAB_SIZE = 50  # number of visual words


def build_vocabulary(descriptor_lists, vocab_size=VOCAB_SIZE, random_state=42):
    # ORB descriptors are binary, but k-means over their float cast is the
    # standard cheap approximation for building a BoVW vocabulary.
    all_descriptors = np.concatenate([d for d in descriptor_lists if d is not None])
    kmeans = MiniBatchKMeans(n_clusters=vocab_size, random_state=random_state, n_init=10)
    kmeans.fit(all_descriptors.astype(np.float32))
    return kmeans


def bovw_histogram(descriptors, vocab):
    # No keypoints found (e.g. a fully masked-out leaf) -> the zero vector.
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(vocab.n_clusters)

    words = vocab.predict(descriptors.astype(np.float32))
    histogram, _ = np.histogram(words, bins=np.arange(vocab.n_clusters + 1))
    return histogram / histogram.sum()
