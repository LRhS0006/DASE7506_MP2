"""Frozen 14-feature verifier. Inference uses only NumPy and supplied bank inputs."""
from pathlib import Path
import numpy as np
from common import parse_answer
from comparison import features as comparison_features, choose
from arithmetic_checker import check_text
from surface import style

FEATURE_NAMES = [
    'own_score', 'other_same_answer_votes', 'supporter_mean_minus_own',
    'best_competitor_max_minus_own', 'supporter_mean_minus_best_competitor_mean',
    'votes_times_own_minus_best_competitor_max',
    'has_contradiction', 'log1p_contradictions',
    'log1p_characters', 'log1p_words', 'newline_density', 'markup_density',
    'digit_density', 'confident_word_rate',
]


def build_features(bank, answers, base_scores):
    contradictions = np.zeros((len(bank), 4, 2), dtype=np.float64)
    for i, question in enumerate(bank):
        for j, candidate in enumerate(question['candidates']):
            count = sum(item['contradiction'] for item in check_text(candidate['text']))
            contradictions[i, j] = [count > 0, np.log1p(count)]
    return np.concatenate([comparison_features(base_scores, answers), contradictions, style(bank)], axis=-1)


def linear_scores(inputs, weights, prefix):
    # StandardScaler preserves float32 embedding inputs. Keep the same two
    # in-place rounding operations as the frozen sklearn implementation.
    transformed = np.array(inputs, copy=True)
    transformed -= weights[prefix + '_mean']
    transformed /= weights[prefix + '_scale']
    return transformed @ weights[prefix + '_coef'].T + weights[prefix + '_intercept']


def predict(bank, arrays, checkpoint):
    if checkpoint is None:
        raise ValueError('Provide the released checkpoint directory')
    if not bank:
        return []
    path = Path(checkpoint)
    if path.is_dir():
        path = path / 'model.npz'
    answers = [[parse_answer(c['text']) for c in q['candidates']] for q in bank]
    with np.load(path, allow_pickle=False) as weights:
        base = linear_scores(arrays['embeddings'].reshape(-1, 1024), weights, 'base').reshape(-1, 4)
        features = build_features(bank, answers, base)
        scores = linear_scores(features.reshape(-1, 14), weights, 'meta').reshape(-1, 4)
    # No question is removed. All-unparseable inputs deterministically use argmax.
    return choose(scores, answers).tolist()
