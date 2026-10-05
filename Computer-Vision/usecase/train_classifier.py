import os
import cv2 as cv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

from load_dataset import SPLITS
from features import extract_features
from orb_detect import make_orb_detector, detect_orb_keypoints
from orb_features import build_vocabulary, bovw_histogram

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.joblib')
VOCAB_PATH = os.path.join(os.path.dirname(__file__), 'orb_vocab.joblib')


def collect_samples(split_path, detector):
    # One image read per sample: base color/texture features and ORB
    # descriptors are both needed, and descriptors additionally feed the
    # shared vocabulary before they can become a BoVW histogram.
    samples = []
    classes = sorted(os.listdir(split_path))

    for label in classes:
        class_dir = os.path.join(split_path, label)
        if not os.path.isdir(class_dir):
            continue
        for filename in os.listdir(class_dir):
            image_path = os.path.join(class_dir, filename)
            image = cv.imread(image_path)
            if image is None:
                continue
            base_features = extract_features(image)
            _, _, _, descriptors = detect_orb_keypoints(image, detector)
            samples.append((base_features, descriptors, label))

    return samples


def build_feature_matrix(samples, vocab):
    features = []
    labels = []
    for base_features, descriptors, label in samples:
        bovw = bovw_histogram(descriptors, vocab)
        features.append(np.concatenate([base_features, bovw]))
        labels.append(label)

    return np.array(features), np.array(labels)


if __name__ == '__main__':
    detector = make_orb_detector()

    print('Reading images and extracting base features + ORB descriptors...')
    train_samples = collect_samples(SPLITS['train'], detector)
    val_samples = collect_samples(SPLITS['validation'], detector)

    print('Building ORB visual vocabulary from training descriptors...')
    train_descriptors = [descriptors for _, descriptors, _ in train_samples]
    vocab = build_vocabulary(train_descriptors)
    joblib.dump(vocab, VOCAB_PATH)
    print(f'Saved ORB vocabulary to {VOCAB_PATH}')

    X_train, y_train = build_feature_matrix(train_samples, vocab)
    X_val, y_val = build_feature_matrix(val_samples, vocab)
    print(f'train: X={X_train.shape} y={y_train.shape}')
    print(f'validation: X={X_val.shape} y={y_val.shape}')

    clf = RandomForestClassifier(n_estimators=300, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    print('\nValidation report:')
    print(classification_report(y_val, y_pred))
    print('Confusion matrix (rows=true, cols=pred):')
    labels = sorted(set(y_val))
    print(labels)
    print(confusion_matrix(y_val, y_pred, labels=labels))

    joblib.dump(clf, MODEL_PATH)
    print(f'\nSaved model to {MODEL_PATH}')
