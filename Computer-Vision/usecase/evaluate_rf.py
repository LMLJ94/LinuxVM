"""Evaluate the classical pipeline (hand-made features + Random Forest) on the
test set, so it can be compared with the transfer-learning results in
../pytorch-quickstart/README.md.

Three feature sets are compared (ablation):
  colour_texture      the 13 features from features.py
  colour_texture_orb  + 50-word ORB bag-of-visual-words (= train_classifier.py)
  orb_only            the ORB histogram alone

Same protocol as the CNN experiments: 5 seeds, an image counts as an error if
the majority-vote prediction is wrong, and per-image comparison with the CNN
errors (exact McNemar test on the images the two models disagree on).
"""
import json
import math
import os
from collections import Counter
from multiprocessing import Pool

import cv2 as cv
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from load_dataset import SPLITS
from features import extract_features, FEATURE_NAMES, FEATURE_SHAPE
from orb_detect import make_orb_detector, detect_orb_keypoints, ORB_N_FEATURES
from orb_features import build_vocabulary, bovw_histogram, VOCAB_SIZE

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(HERE, 'rf_feature_cache.joblib')
RESULTS_PATH = os.path.join(HERE, 'rf_results.json')
CNN_RESULTS_PATH = os.path.join(HERE, '..', 'pytorch-quickstart', 'experiment_variants_results.json')

SEEDS = [0, 1, 2, 3, 4]
N_TREES = 300
VOCAB_SEED = 42

# Everything the cached features depend on. A cache made with other settings
# is refused rather than silently reused.
RECIPE = {
    'feature_names': FEATURE_NAMES,
    'feature_shape': list(FEATURE_SHAPE),
    'orb_n_features': ORB_N_FEATURES,
}

CNN_REFERENCES = [
    ('resnet18', 'baseline', 'ResNet18'),
    ('yolov8n_cls', 'baseline', 'YOLOv8n'),
    ('yolov8n_cls', 'clahe', 'YOLOv8n + CLAHE'),
    ('vit_b16_google', 'baseline', 'ViT-B/16'),
]


def list_split(split_path):
    items = []
    for label in sorted(os.listdir(split_path)):
        class_dir = os.path.join(split_path, label)
        if not os.path.isdir(class_dir):
            continue
        for filename in sorted(os.listdir(class_dir)):
            items.append((os.path.join(class_dir, filename), label))
    return items


def _extract_one(item):
    path, label = item
    image = cv.imread(path)
    if image is None:
        return None
    detector = make_orb_detector()
    _, _, _, descriptors = detect_orb_keypoints(image, detector)
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem, label, extract_features(image), descriptors


def load_or_extract():
    if os.path.exists(CACHE_PATH):
        cache = joblib.load(CACHE_PATH)
        if cache.get('recipe') != RECIPE:
            raise RuntimeError(f'{CACHE_PATH} was made with a different feature recipe; delete it to re-extract')
        print(f'Loaded cached features from {CACHE_PATH}')
        return cache['splits']

    splits = {}
    with Pool(os.cpu_count()) as pool:
        for name in ['train', 'validation', 'test']:
            items = list_split(SPLITS[name])
            print(f'Extracting {name}: {len(items)} images...')
            samples = [s for s in pool.map(_extract_one, items, chunksize=8) if s is not None]
            splits[name] = samples
    joblib.dump({'recipe': RECIPE, 'splits': splits}, CACHE_PATH)
    print(f'Saved features to {CACHE_PATH}')
    return splits


def build_matrices(splits, vocab):
    out = {}
    for name, samples in splits.items():
        base = np.array([s[2] for s in samples])
        orb = np.array([bovw_histogram(s[3], vocab) for s in samples])
        out[name] = {
            'stems': [s[0] for s in samples],
            'labels': np.array([s[1] for s in samples]),
            'colour_texture': base,
            'colour_texture_orb': np.hstack([base, orb]),
            'orb_only': orb,
        }
    return out


def column_names(variant):
    orb_names = [f'orb_word_{i}' for i in range(VOCAB_SIZE)]
    return {
        'colour_texture': FEATURE_NAMES,
        'colour_texture_orb': FEATURE_NAMES + orb_names,
        'orb_only': orb_names,
    }[variant]


def mcnemar_exact(b, c):
    # Two-sided exact binomial test on the discordant pairs.
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def run_variant(data, variant):
    train, val, test = data['train'], data['validation'], data['test']
    classes = sorted(set(train['labels']))
    test_preds, val_accs, test_accs, importances = [], [], [], []

    for seed in SEEDS:
        clf = RandomForestClassifier(n_estimators=N_TREES, random_state=seed, n_jobs=-1)
        clf.fit(train[variant], train['labels'])
        val_accs.append(float((clf.predict(val[variant]) == val['labels']).mean()))
        pred = clf.predict(test[variant])
        test_accs.append(float((pred == test['labels']).mean()))
        test_preds.append(pred)
        importances.append(clf.feature_importances_)

    test_preds = np.array(test_preds)  # (seeds, images)
    majority = np.array([Counter(col).most_common(1)[0][0] for col in test_preds.T])

    errors = {}
    for i, stem in enumerate(test['stems']):
        actual = test['labels'][i]
        if majority[i] != actual:
            errors[stem] = {
                'actual': str(actual),
                'predicted': str(majority[i]),
                'wrong_in_seeds': int((test_preds[:, i] != actual).sum()),
            }

    report = classification_report(test['labels'], majority, labels=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(test['labels'], majority, labels=classes)
    mean_imp = np.mean(importances, axis=0)
    names = column_names(variant)
    ranked = sorted(zip(names, mean_imp), key=lambda t: -t[1])

    return {
        'classes': classes,
        'val_acc_mean': float(np.mean(val_accs)),
        'test_acc_per_seed': test_accs,
        'test_acc_mean': float(np.mean(test_accs)),
        'test_acc_majority': float((majority == test['labels']).mean()),
        'per_class': {c: {'precision': report[c]['precision'], 'recall': report[c]['recall']} for c in classes},
        'confusion_matrix': cm.tolist(),
        'errors': errors,
        'importance_top': [(n, float(v)) for n, v in ranked[:10]],
        'importance_orb_share': float(sum(v for n, v in zip(names, mean_imp) if n.startswith('orb_'))),
    }


def compare_with_cnns(rf_errors, test_stems):
    if not os.path.exists(CNN_RESULTS_PATH):
        print(f'\n(no CNN results at {CNN_RESULTS_PATH}; skipping comparison)')
        return {}
    cnn = json.load(open(CNN_RESULTS_PATH))
    rf_wrong = set(rf_errors)
    out = {}
    for backbone, variant, label in CNN_REFERENCES:
        entry = cnn.get(backbone, {}).get(variant)
        if entry is None:
            continue
        if entry['n_test'] != len(test_stems):
            raise RuntimeError(f'{label}: CNN test set has {entry["n_test"]} images, RF has {len(test_stems)}')
        cnn_wrong = set(entry['errors'])
        only_rf = len(rf_wrong - cnn_wrong)   # CNN right, RF wrong
        only_cnn = len(cnn_wrong - rf_wrong)  # RF right, CNN wrong
        out[label] = {
            'cnn_errors': len(cnn_wrong),
            'both_wrong': sorted(rf_wrong & cnn_wrong),
            'rf_wrong_cnn_right': only_rf,
            'cnn_wrong_rf_right': only_cnn,
            'mcnemar_p': mcnemar_exact(only_rf, only_cnn),
        }
    return out


def print_variant(name, r):
    print(f'\n=== {name} ===')
    accs = r['test_acc_per_seed']
    print(f'Test {r["test_acc_mean"]:.1%} (min {min(accs):.1%}, max {max(accs):.1%}), '
          f'majority vote {r["test_acc_majority"]:.1%}, '
          f'{len(r["errors"])} errors; validation {r["val_acc_mean"]:.1%}')
    print('Per class (majority vote):  precision  recall')
    for c, m in r['per_class'].items():
        print(f'  {c:<10} {m["precision"]:>17.1%} {m["recall"]:>7.1%}')
    print('Confusion matrix (rows = actual, cols = predicted):', r['classes'])
    for c, row in zip(r['classes'], r['confusion_matrix']):
        print(f'  {c:<10} {row}')
    print(f'ORB share of feature importance: {r["importance_orb_share"]:.1%}')
    print('Top features:', ', '.join(f'{n} {v:.3f}' for n, v in r['importance_top'][:6]))


if __name__ == '__main__':
    splits = load_or_extract()

    print('Building ORB vocabulary from training descriptors...')
    vocab = build_vocabulary([s[3] for s in splits['train']], random_state=VOCAB_SEED)
    data = build_matrices(splits, vocab)
    for name in ['train', 'validation', 'test']:
        print(f'{name}: {len(data[name]["stems"])} images')

    results = {}
    for variant in ['colour_texture', 'colour_texture_orb', 'orb_only']:
        results[variant] = run_variant(data, variant)
        print_variant(variant, results[variant])

    # ORB turned out to lower accuracy, so both RF variants are compared.
    results['comparison_with_cnns'] = {}
    for variant in ['colour_texture', 'colour_texture_orb']:
        print(f'\n=== {variant} vs. CNNs (per image) ===')
        comparison = compare_with_cnns(results[variant]['errors'], data['test']['stems'])
        for label, c in comparison.items():
            print(f'  {label:<16} CNN errors {c["cnn_errors"]}, both wrong {len(c["both_wrong"])}, '
                  f'RF wrong/CNN right {c["rf_wrong_cnn_right"]}, CNN wrong/RF right {c["cnn_wrong_rf_right"]}, '
                  f'McNemar p = {c["mcnemar_p"]:.2g}')
            if c['both_wrong']:
                print(f'    both wrong: {", ".join(s[:8] for s in c["both_wrong"])}')
        results['comparison_with_cnns'][variant] = comparison

    ct, cto = set(results['colour_texture']['errors']), set(results['colour_texture_orb']['errors'])
    p = mcnemar_exact(len(ct - cto), len(cto - ct))
    print(f'\nAdding ORB: fixes {len(ct - cto)}, breaks {len(cto - ct)} (McNemar p = {p:.2g})')
    results['orb_ablation'] = {'fixed': len(ct - cto), 'broken': len(cto - ct), 'mcnemar_p': p}

    print('\nRF errors (colour_texture):')
    for stem, e in results['colour_texture']['errors'].items():
        print(f'  {stem[:8]}  {e["actual"]:<8} -> {e["predicted"]:<8} wrong in {e["wrong_in_seeds"]}/5')

    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved results to {RESULTS_PATH}')
