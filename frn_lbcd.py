# -*- coding: utf-8 -*-
import os
import random
import numpy as np
from copy import deepcopy


def select_neighbours(p, node):

    """
    :param p: N*N adjacency matrix
    :param node: index of a node
    :return: a shuffled list of neighbour nodes
    """
    neighbors = np.where(p[node, :] > 0)[0].tolist()
    neighbors = random.sample(neighbors, len(neighbors))
    return neighbors


def compute_gain(y, p, rho, xi, indices):

    cv_y = np.sum(np.power(np.mean(y[indices, :], axis=0), 2))  # Cv in the criterion
    ss_y = np.sum([p[i, j] for i in indices for j in indices])  # summary similarities
    Lambda = np.divide(ss_y, len(indices) ** 2)
    gain = rho * cv_y * len(indices) + xi * Lambda * ss_y

    return gain


def phase_one(y, p, labels, rho, xi, greedy=False):

    """
    :param y: N*V entity-to-feature matrix
    :param p: N*N adjacency matrix
    :param labels: dict where keys represents the nodes index and values represents the corresponding community label
    :param rho:
    :param xi:
    :param greedy:
    :return: list of sets, each list represents the indices of nodes of a cluster (updated version of input labels)
    """

    for node, label in labels.items():

        print("node:", node, "current community:", label)

        gain_old = compute_gain(y=y, p=p, rho=rho, xi=xi, indices=[node])

        print("old gain:", gain_old)

        neighbours = select_neighbours(p=p, node=node)
        print("neighbours:", neighbours)

        if greedy is True:

            tmp_gains = np.zeros([y.shape[0]])

        for neighbour in neighbours:

            print("neighbour:", neighbour)

            gain_new = compute_gain(y=y, p=p, rho=rho, xi=xi, indices=[node, neighbour])
            print("new gain:", gain_new)
            delta = gain_new - gain_old

            if greedy is True and delta > 0:
                tmp_gains[neighbour] = delta
            elif greedy is False and delta > 0:
                labels[node] = labels[neighbour]
                break

    print("labels at the end of 1st stage:", labels)

    return labels


def phase_two(y, p, labels):

    clusters_labels = list(set(labels.values()))
    clusters_indices = {label: [] for label in clusters_labels}
    clusters_number = len(clusters_labels)

    y_agg = np.zeros([clusters_number, y.shape[1]])
    p_agg = np.zeros([clusters_number, clusters_number])

    clusters_mean = {}
    clusters_summary_similarity = {}
    updated_labels = {}

    # grouping clusters w.r.t their indices
    for k, v in labels.items():
        for label in range(len(clusters_labels)):
            if v == clusters_labels[label]:
                clusters_indices[clusters_labels[label]].append(k)
                updated_labels[k] = label

    print("updated_labels:", updated_labels)

    # computing the average and summary similarity for moving the community centers
    for label, indices in clusters_indices.items():
        clusters_mean[label] = np.mean(y[indices, :], axis=0)
        clusters_summary_similarity[label] = np.sum([p[i, j] for i in indices for j in indices])

    print("clusters_mean:", clusters_mean)
    print("clusters_summary_similarity:", clusters_summary_similarity)

    # Aggregation
    key = 0
    for k, v in clusters_mean.items():
        print("c:", v, "s:",  clusters_summary_similarity[k])
        y_agg[key, :] = v
        p_agg[key, key] = clusters_summary_similarity[k]
        key += 1

    return y_agg, p_agg, updated_labels


def louvain(y, p, labels, rho, xi):

    f1 = True

    while f1 is True:

        labels_old = deepcopy(labels)

        labels_new = phase_one(y=y, p=p, labels=labels, rho=rho, xi=xi)
        print("labels after 1st phase:", labels_new)

        cntr = 0

        print("lables old:", labels_old)

        for k, v in labels_new.items():
            if v != labels_old[k]:
                print("inja")
                cntr = 0
                break
            elif v == labels_old[k]:
                print("onja")
                cntr += 1

        if cntr == y.shape[0]:
            f1 = False
        else:
            labels = deepcopy(labels_new)

    y_agg, p_agg, updated_labels = phase_two(y=y, p=p, labels=labels_new)

    return labels


if __name__ == '__main__':

    y = np.array([[1], [1], [1], [2], [2], [2]])

    p = np.array([[0, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 1, 1], [0, 0, 0, 1, 0, 1], [0, 0, 0, 1, 1, 0]])

    labels = {k: k+1 for k in range(y.shape[0])}

    print("initial labels:", labels)

    final_labels = louvain(y=y, p=p, labels=labels, rho=1, xi=1)

    print("final_labels:", final_labels)