import random
from copy import deepcopy
from sklearn.metrics import confusion_matrix

import random
from collections import defaultdict
import math

random.seed(42)
import numpy as np


def dict2list(input_dict):
    out = np.zeros(len(input_dict), dtype=int)
    for index, value in input_dict.items():
        out[index] = value
    return out


def list_mean(list_):
    sum_ = 0
    for ele in list_:
        sum_ += ele
    return sum_ * 1.0 / len(list_)


def dict_values_allclose(dict1, dict2, rtol=1e-3, atol=1e-03):
    for item in dict1.keys():
        for class_ in dict1[item].keys():
            if math.fabs(dict1[item][class_] - dict2[item][class_]) > atol + rtol * math.fabs(dict2[item][class_]):
                return False
    return True


def iwmv(e2wl, w2el, label_set, its_=10, T_required=False):
    reliabilities = []
    truths = {}
    votes = {}
    v = dict()
    v = defaultdict(lambda: 1, v)
    worker_num_correct = {}
    worker_num_correct = defaultdict(lambda: 0, worker_num_correct)
    for it in range(its_):
        worker_num_correct.clear()
        # update truth
        for item in e2wl.keys():
            item_votes = {}
            for class_ in label_set:
                item_votes[class_] = 0
            for worker, label in e2wl[item]:
                for class_ in label_set:
                    if label == class_:
                        item_votes[class_] += v[worker]
            truths[item] = extract_truth_from_dict(item_votes)
            for worker, label in e2wl[item]:
                if label == truths[item]:
                    worker_num_correct[worker] += 1
            votes[item] = item_votes

        # if len(prev_votes) != 0 and dict_values_allclose(prev_votes, votes):
        #     break

        # update worker ability
        for worker, worker_labels in w2el.items():
            if it == its_ - 1:
                reliabilities.append((worker_num_correct[worker] / len(worker_labels)))
            v[worker] = len(label_set) * (worker_num_correct[worker] / len(worker_labels)) - 1

    T_matrix = generate_T_matrix(w2el, truths, label_set, v)
    if T_required:
        return truths, it + 1, T_matrix
    else:
        return truths, it + 1


def extract_truth_from_dict(dict_):
    max_ = -9999
    candidate = []
    for class_ in dict_.keys():
        val = dict_[class_]
        if val > max_:
            max_ = val
            candidate.clear()
            candidate.append(class_)
        elif val == max_:
            candidate.append(class_)
        else:
            continue
    return random.choice(candidate)


def gete2wlandw2el(media=None):
    e2wl = {}
    w2el = {}
    label_set = []
    results = []
    for item, single_annotation in enumerate(media):
        for worker, label in enumerate(single_annotation):
            label_set.append(label)
            results.append({"task": item, "worker": worker, "label": label})
            if item not in e2wl:
                e2wl[item] = []
            e2wl[item].append([worker, label])

            if worker not in w2el:
                w2el[worker] = []
            w2el[worker].append([item, label])
    maximum = max(label_set)
    label_set = [x for x in range(0, maximum + 1)]
    return e2wl, w2el, label_set


def generate_T_matrix(w2el, truths, label_set, v) -> np.array:
    t_first_dict = {}
    for worker, list_with_annotations in w2el.items():
        t_first_dict[worker] = np.zeros((len(label_set), len(label_set)))
        for single_list in list_with_annotations:
            sample = single_list[0]
            worker_annotations = single_list[1]
            t_first_dict[worker][int(truths[sample])][int(worker_annotations)] += 1

        row_sums = t_first_dict[worker].sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        t_first_dict[worker] = t_first_dict[worker] / row_sums

    final_T = np.zeros((len(label_set), len(label_set)))

    for annotator_id, matrix in t_first_dict.items():
        reliability = v[annotator_id]
        final_T += matrix * reliability
    row_sums = final_T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    final_T = final_T / row_sums
    row_sums = final_T.sum(axis=1)
    # assert np.allclose(row_sums, 1)
    return final_T
