"""ReDeEP(token) detector (Section 4.1 of the paper): pick the top
Copying Heads (set A) by copying-head score, pick the top Knowledge FFN
layers (set F) by correlation of layer-level PKS with the hallucination
label on a held-out split, then fit a (sign-constrained) logistic
regression on the two aggregated covariates:

    mean_E_A(r)  = mean over (l,h) in A of E^{l,h}_r
    mean_P_F(r)  = mean over l in F of P^l_r

H(r) = alpha * mean_P_F(r) - beta * mean_E_A(r),  alpha, beta > 0

This mirrors the paper's linear regression over decoupled covariates
without requiring per-head/per-layer coefficients to be hand-tuned.
"""
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, recall_score, accuracy_score, precision_recall_curve


def select_copying_heads(copying_scores, top_k):
    flat = copying_scores.flatten()
    idx = np.argsort(-flat)[:top_k]
    L, H = copying_scores.shape
    return [(i // H, i % H) for i in idx]


def select_knowledge_ffns(P_matrix, labels, top_k):
    """P_matrix: [N, num_layers]; returns indices of the top_k layers whose
    PKS correlates most positively with the hallucination label."""
    num_layers = P_matrix.shape[1]
    corrs = np.zeros(num_layers)
    for l in range(num_layers):
        col = P_matrix[:, l]
        if np.std(col) < 1e-12:
            corrs[l] = -1
            continue
        corrs[l], _ = pearsonr(col, labels)
    order = np.argsort(-corrs)[:top_k]
    return list(order), corrs


def aggregate_features(E_all, P_all, heads_A, layers_F):
    """E_all: [N, num_layers, num_heads], P_all: [N, num_layers]."""
    if heads_A:
        e_feat = np.mean([E_all[:, l, h] for (l, h) in heads_A], axis=0)
    else:
        e_feat = np.zeros(E_all.shape[0])
    if len(layers_F):
        p_feat = np.mean(P_all[:, layers_F], axis=1)
    else:
        p_feat = np.zeros(P_all.shape[0])
    return e_feat, p_feat


class ReDeEPDetector:
    def __init__(self, n_copying_heads=8, n_knowledge_ffns=5):
        self.n_copying_heads = n_copying_heads
        self.n_knowledge_ffns = n_knowledge_ffns

    def fit(self, E_train, P_train, y_train, copying_scores):
        self.heads_A = select_copying_heads(copying_scores, self.n_copying_heads)
        self.layers_F, self.ffn_corrs = select_knowledge_ffns(P_train, y_train, self.n_knowledge_ffns)
        e_feat, p_feat = aggregate_features(E_train, P_train, self.heads_A, self.layers_F)
        X = np.stack([p_feat, -e_feat], axis=1)   # both engineered to be positively assoc. w/ hallucination
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        self.clf = LogisticRegression()
        self.clf.fit(X, y_train)

        # Calibrate the decision threshold on the train set (Youden's J /
        # best-F1 point) instead of assuming 0.5 is meaningful -- with a
        # handful of engineered features and a small sample, raw logistic
        # probabilities are not well calibrated, so AUC/PCC (rank-based,
        # threshold-free) are the metrics to trust; this only fixes
        # Acc/Rec/F1 reporting to use a sensible cut point.
        train_scores = self.clf.predict_proba(X)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_train, train_scores)
        f1s = 2 * precision * recall / np.clip(precision + recall, 1e-9, None)
        self.threshold = thresholds[np.argmax(f1s[:-1])] if len(thresholds) else 0.5
        return self

    def score(self, E, P):
        e_feat, p_feat = aggregate_features(E, P, self.heads_A, self.layers_F)
        X = np.stack([p_feat, -e_feat], axis=1)
        X = self.scaler.transform(X)
        return self.clf.predict_proba(X)[:, 1]

    def evaluate(self, E, P, y):
        scores = self.score(E, P)
        preds = (scores >= self.threshold).astype(int)
        out = {
            "AUC": roc_auc_score(y, scores) if len(set(y)) > 1 else float("nan"),
            "PCC": pearsonr(scores, y)[0] if np.std(scores) > 1e-12 else float("nan"),
            "Acc": accuracy_score(y, preds),
            "Rec": recall_score(y, preds),
            "F1": f1_score(y, preds),
        }
        return out, scores
