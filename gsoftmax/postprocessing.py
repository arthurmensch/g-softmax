import joblib
import torch
import numpy as np
from gsoftmax.sinkhorn import pairwise_distance


def pairwise_l2_dist(p):
    D = p.unsqueeze(2) - p.unsqueeze(1)
    return ((D * D).sum(-1).clamp(min=0.)).sqrt()


def sum_pow_mat(A, max_n=15):
    S = A.clone()
    for i in range(max_n):  # lots of allocation...
        S += S.matmul(A)
        S.sign_()
    return S


def cluster_and_trim(positions, weights, pre_weight_thresh, distance_threshold,
                     post_weight_thresh, n_steps=10):
    weights = weights.clone()
    weights[weights < pre_weight_thresh] = 0.0

    if distance_threshold == 0:
        return positions, weights

    A = (- pairwise_distance(positions, positions, p=2, q=1) < distance_threshold).float()
    # zero out rows and columns corresponding to zero-weight positions.
    zero_inds = ((weights == 0).unsqueeze(-1).float() * (
        torch.ones_like(weights).unsqueeze(-2))).sign_().byte()
    A[zero_inds] = 0
    zero_inds = ((weights == 0).unsqueeze(-2).float() * (
        torch.ones_like(weights).unsqueeze(-1))).sign_().byte()
    A[zero_inds] = 0
    # Compute connected components using power iteration
    C = sum_pow_mat(A, n_steps)
    weighted_C = C * weights.unsqueeze(-2)
    C_weights = weighted_C.sum(-1)
    C_means = (weighted_C.matmul(positions))
    C_means[C_weights != 0] /= C_weights[C_weights != 0].unsqueeze(-1)

    C_weights[C_weights < post_weight_thresh] = 0.0

    proc_positions = torch.zeros_like(positions)
    proc_weights = torch.zeros_like(weights)

    C = C.cpu().numpy()
    for b in range(positions.shape[0]):
        _, inds = np.unique(C[b], axis=0, return_index=True)
        proc_positions[b, :len(inds)] = C_means[b, inds]
        proc_weights[b, :len(inds)] = C_weights[b, inds]
    return proc_positions, proc_weights
