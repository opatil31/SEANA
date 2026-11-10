"""ANOVA variance decomposition, two distinct ranks, and cosine retrieval.

Imports torch only when operating on tensors, so captured reports need only Python.
"""

import math


def anova_from_moments(s_w, s_b, n_images, k_reps):
    """Trace-pooled one-way ANOVA for equally sized clean-image identity groups.

    s_w = mean_i mean_k ||f_ik - mean_k f_ik||^2
    s_b = mean_i ||mean_k f_ik - grand_mean||^2
    The historical ndv is 1-eta^2, not the usual variance-component ICC(1,1).
    """
    if n_images < 2 or k_reps < 2:
        raise ValueError("ANOVA requires at least two images and two noise repetitions")
    if not all(math.isfinite(v) and v >= 0 for v in (s_w, s_b)):
        raise ValueError("Variances must be finite and nonnegative")
    ss_within = n_images * k_reps * s_w
    ss_between = n_images * k_reps * s_b
    ms_within = ss_within / (n_images * (k_reps - 1))
    ms_between = ss_between / (n_images - 1)
    total = s_w + s_b
    icc_denominator = ms_between + (k_reps - 1) * ms_within
    icc = (ms_between - ms_within) / icc_denominator if icc_denominator > 0 else None
    return {
        "s_w": s_w,
        "s_b": s_b,
        "s_total": total,
        "ss_within": ss_within,
        "ss_between": ss_between,
        "df_within": n_images * (k_reps - 1),
        "df_between": n_images - 1,
        "ms_within": ms_within,
        "ms_between": ms_between,
        "noise_variance_fraction": s_w / total if total > 0 else None,
        "eta_squared": s_b / total if total > 0 else None,
        "icc_1_1_trace": icc,
        "one_minus_icc_1_1_trace": 1 - icc if icc is not None else None,
        "within_variance_component": ms_within,
        "between_variance_component": (ms_between - ms_within) / k_reps,
        "variance_status": "defined" if total > 0 else "constant_embeddings",
    }


def covariance_ranks(image_means):
    """Ranks of the covariance of per-image mean embeddings, not all N*K samples."""
    import torch

    x = image_means.detach().double().cpu()
    if x.ndim != 2 or len(x) < 2 or x.shape[1] < 1 or not torch.isfinite(x).all():
        raise ValueError("Expected finite [N,D] image means, with N >= 2 and D >= 1")
    x = x - x.mean(0, keepdim=True)
    # The smaller Gram matrix has the same nonzero spectrum as feature covariance.
    gram = (x @ x.T if len(x) < x.shape[1] else x.T @ x) / (len(x) - 1)
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0)
    total = eigenvalues.sum()
    if total <= 0:
        return {"effective_rank_entropy": 0.0, "participation_ratio": 0.0}
    probabilities = eigenvalues / total
    participation_ratio = float(1 / probabilities.square().sum())
    positive = probabilities[probabilities > 1e-12]  # threshold in the captured implementation
    entropy_rank = float(torch.exp(-(positive * positive.log()).sum()))
    return {"effective_rank_entropy": entropy_rank, "participation_ratio": participation_ratio}


def embedding_statistics(features):
    """features[N,K,D] contains raw backbone features; labels are never used."""
    import torch

    x = features.detach().double().cpu()
    if x.ndim != 3 or x.shape[0] < 2 or x.shape[1] < 2 or x.shape[2] < 1:
        raise ValueError("Expected [N,K,D] with N,K >= 2 and D >= 1")
    if not torch.isfinite(x).all():
        raise ValueError("Nonfinite backbone embeddings")
    means = x.mean(1)
    s_w = float((x - means[:, None]).square().sum(-1).mean())
    s_b = float((means - means.mean(0)).square().sum(-1).mean())
    return {**anova_from_moments(s_w, s_b, x.shape[0], x.shape[1]), **covariance_ranks(means)}


def retrieval_scores(queries, gallery, *, chunk_size=256, permutation_seed=0):
    """Cosine top-1 matching of corresponding rows, plus one shuffled-label control.

    First-index argmax breaks ties, matching the captured script. Zero vectors
    are allowed (collapse is part of this analysis); tied matches are not evidence
    of preserved content. The permutation result is a control, not a p-value.
    """
    import torch
    from torch.nn.functional import normalize

    q, g = queries.detach().float().cpu(), gallery.detach().float().cpu()
    if q.ndim != 2 or q.shape != g.shape or len(q) < 2 or q.shape[1] < 1:
        raise ValueError("Retrieval requires matching [M,D] matrices with M >= 2")
    if chunk_size < 1 or not torch.isfinite(q).all() or not torch.isfinite(g).all():
        raise ValueError("Invalid retrieval chunk size or nonfinite features")
    q, g = normalize(q, dim=1), normalize(g, dim=1)
    predictions = torch.cat([(q[i : i + chunk_size] @ g.T).argmax(1) for i in range(0, len(q), chunk_size)])
    labels = torch.arange(len(q))
    shuffled_labels = torch.randperm(len(q), generator=torch.Generator().manual_seed(permutation_seed))
    return {
        "top1": float((predictions == labels).double().mean()),
        "permuted_top1": float((predictions == shuffled_labels).double().mean()),
        "chance": 1 / len(q),
    }
