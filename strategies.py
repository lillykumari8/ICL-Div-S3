import os
import random
import json
import torch
import time
import submarine
import numpy as np
from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import euclidean_distances, pairwise, pairwise_distances
from utils import codex_execution


def compute_sims(data, sim_metric="cosine", sigma=1.0):
    if sim_metric == "cosine":
        sims = cosine_similarity(data).astype(np.float32)
        sims[sims < 0] = 0
        return sims
    elif sim_metric == "cos1":
        sims = cosine_similarity(data).astype(np.float32)
        sims += 1
        return sims
    else:
        if sim_metric == "ham":
            data = data > 0
            metric_name = "hamming"
        elif sim_metric == "corr":
            metric_name = "correlation"
        elif sim_metric == "maha":
            metric_name = "mahalanobis"
        elif sim_metric == "jac":
            data = data > 0
            metric_name = "jaccard"
        elif sim_metric == "l2":
            metric_name = "euclidean"

        if sim_metric == "sq_l2":
            dists = euclidean_distances(data, squared=True).astype(np.float32)
        else:
            dists = pairwise_distances(data, metric=metric_name, n_jobs=16).astype(
                np.float32
            )
        for ii in range(data.shape[0]):
            dists[ii, ii] = 0
        sims = np.exp(-dists / sigma).astype(np.float32)
        del dists
        return sims


def compute_sims_matrix_vector(data, feat_vec, sim_metric="cosine", sigma=1.0):

    if sim_metric == "cosine":
        sims = cosine_similarity(feat_vec.reshape(1, -1), data).astype(np.float32)
        sims[sims < 0] = 0
        return sims
    elif sim_metric == "cos1":
        sims = cosine_similarity(feat_vec.reshape(1, -1), data).astype(np.float32)
        sims += 1
        return sims
    else:
        if sim_metric == "ham":
            data = data > 0
            metric_name = "hamming"
        elif sim_metric == "corr":
            metric_name = "correlation"
        elif sim_metric == "maha":
            metric_name = "mahalanobis"
        elif sim_metric == "jac":
            data = data > 0
            metric_name = "jaccard"
        elif sim_metric == "l2":
            metric_name = "euclidean"

        if sim_metric == "sq_l2":
            dists = euclidean_distances(
                feat_vec.reshape(1, -1), data, squared=True
            ).astype(np.float32)
        else:
            dists = pairwise_distances(
                feat_vec.reshape(1, -1), data, metric=metric_name, n_jobs=16
            ).astype(np.float32)

        sims = np.exp(-dists / sigma).astype(np.float32)
        del dists
        return sims


def fast_votek(embeddings, select_num, k, vote_file=None):
    n = len(embeddings)
    if vote_file is not None and os.path.isfile(vote_file):
        with open(vote_file) as f:
            vote_stat = json.load(f)
    else:
        bar = tqdm(range(n), desc=f"voting")
        vote_stat = defaultdict(list)
        for i in range(n):
            cur_emb = embeddings[i].reshape(1, -1)
            cur_scores = np.sum(cosine_similarity(embeddings, cur_emb), axis=1)
            sorted_indices = np.argsort(cur_scores).tolist()[-k - 1 : -1]
            for idx in sorted_indices:
                if idx != i:
                    vote_stat[idx].append(i)
            bar.update(1)
        if vote_file is not None:
            with open(vote_file, "w") as f:
                json.dump(vote_stat, f)
    votes = sorted(vote_stat.items(), key=lambda x: len(x[1]), reverse=True)
    selected_indices = []
    selected_times = defaultdict(int)
    while len(selected_indices) < select_num:
        cur_scores = defaultdict(int)
        for idx, candidates in votes:
            if idx in selected_indices:
                cur_scores[idx] = -100
                continue
            for one_support in candidates:
                if not one_support in selected_indices:
                    cur_scores[idx] += 10 ** (-selected_times[one_support])
        cur_selected_idx = max(cur_scores.items(), key=lambda x: x[1])[0]
        selected_indices.append(int(cur_selected_idx))
        for idx_support in vote_stat[cur_selected_idx]:
            selected_times[idx_support] += 1
    return selected_indices


def subm_diversity(features, annotation_budget, sim_metric="cosine", sigma=1.0):
    sim_mat = compute_sims(features, sim_metric=sim_metric, sigma=sigma)
    fl = submarine.FacilityLocationFunction(sim_mat)
    sol_order = submarine.VectorInt()
    sol_gains = submarine.VectorFloat()
    sol_set = submarine.DefaultSet(features.shape[0])
    submarine.AcceleratedGreedyConstrainedMaximization(
        fl, annotation_budget, sol_set, sol_order, sol_gains
    )
    fl_set_eval = fl.Evaluate(sol_set)
    soln = list(sol_set.get_elements())
    return soln, fl_set_eval


def modular_min(f, q_set, grnd_set_size, reduced_set=[], mode="faq"):
    assert len(q_set.get_elements()) == 1
    inc_gains = np.full(grnd_set_size, np.inf, dtype="float64")
    if not reduced_set:
        reduced_set = list(
            set([i for i in range(grnd_set_size)]) - set(list(q_set.get_elements()))
        )

    t = f.NewGrowingContextExternal(q_set)
    if mode == "faq":
        inc_gains = [
            t.Gain(i, q_set) if i in reduced_set else inc_gains[i]
            for i in range(grnd_set_size)
        ]
    elif mode == "fqa":
        q = list(q_set.get_elements())[0]
        for idx in reduced_set:
            a_set = submarine.DefaultSet(set([idx]), grnd_set_size)
            inc_gains[idx] = f.Gain(q, a_set)
    inc_inds = np.argsort(inc_gains)
    return list(inc_inds)


def submodular_kc_greedy(sim_mat, cost_overall, kcbudget, kcpower, cost_words):
    fl = submarine.FacilityLocationFunction(sim_mat)
    n = sim_mat.shape[0]
    grnd_set_size = n
    summary_set = submarine.DefaultSet(set(), grnd_set_size)
    comp_set = set([i for i in range(n)])
    cost = 0
    cond = True
    while cond:
        gains = np.array([0.0 if idx in comp_set else -np.inf for idx in range(n)])
        for i, index in enumerate(comp_set):
            gains[index] = fl.Gain(index, summary_set) / np.power(
                cost_overall[index], float(kcpower)
            )
        gains_indices = gains.argsort()[::-1]
        if cost + cost_words[gains_indices[0]] < kcbudget:
            summary_set.insert(gains_indices[0])
            comp_set = comp_set - set([gains_indices[0]])
            cost += cost_words[gains_indices[0]]
            print(
                f"Gain: {gains[gains_indices[0]]}, Cost: {cost_words[gains_indices[0]]}"
            )
        else:
            cond = False
    print(f"fl of summary: {fl.Evaluate(summary_set)}")
    return summary_set


def submodular_greedy(sim_mat, kcbudget, card_budget, cost_words):
    fl = submarine.FacilityLocationFunction(sim_mat)
    n = sim_mat.shape[0]
    grnd_set_size = n
    summary_set = submarine.DefaultSet(set(), grnd_set_size)
    comp_set = set([i for i in range(n)])
    cost = 0
    for i in range(card_budget):
        gains = np.array([0.0 if idx in comp_set else -np.inf for idx in range(n)])
        for i, index in enumerate(comp_set):
            gains[index] = fl.Gain(index, summary_set)
        gains_indices = gains.argsort()[::-1]
        if cost + cost_words[gains_indices[0]] > kcbudget:
            break
        summary_set.insert(gains_indices[0])
        comp_set = comp_set - set([gains_indices[0]])
        cost += cost_words[gains_indices[0]]
        print(f"Gain: {gains[gains_indices[0]]}, Cost: {cost_words[gains_indices[0]]}")
    print(f"fl of summary: {fl.Evaluate(summary_set)}")
    return summary_set


def maximize_diversity(embeddings, annotation_size):
    selected_indices = []
    first_id = random.choice(range(len(embeddings)))
    selected_indices.append(first_id)
    selected_representations = embeddings[first_id].reshape(1, -1)
    for count in range(annotation_size - 1):
        scores = np.sum(cosine_similarity(embeddings, selected_representations), axis=1)
        for i in selected_indices:
            scores[i] = float("inf")
        min_idx = np.argmin(scores)
        selected_representations = torch.cat(
            (selected_representations, embeddings[min_idx].reshape(1, -1)), 0
        )
        selected_indices.append(min_idx.item())
    return selected_indices


def maximize_fl(embeddings, annotation_size):
    N, D = embeddings.shape
    sim_mat = compute_sims(embeddings, sim_metric="cosine")
    fl = submarine.FacilityLocationFunction(sim_mat)

    norm_embeds = embeddings / embeddings.norm(dim=1, keepdim=True)
    cosine = torch.einsum("nd,md->nm", norm_embeds, norm_embeds)
    cosine = torch.relu(cosine)
    selected = torch.zeros(N, dtype=torch.bool)
    max_similarity = torch.zeros(N) - 1
    for k in tqdm(range(annotation_size)):
        marginal_gain = torch.relu(cosine - max_similarity).sum(dim=1) * (
            1 - selected.float()
        )
        node = torch.argmax(marginal_gain)
        selected[node] = True
        max_similarity = torch.max(max_similarity, cosine[node])
    selected_indices = torch.nonzero(selected).squeeze().tolist()
    sel_set = submarine.DefaultSet(set(selected_indices), N)
    fl_set_eval = fl.Evaluate(sel_set)
    return selected_indices, fl_set_eval


def get_instance_length(input_text, output_text, tokenizer):
    return len(tokenizer(input_text)["input_ids"]), len(
        tokenizer(output_text)["input_ids"]
    )
