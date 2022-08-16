# -*- coding: utf-8 -*-
import os
import time
import random
import pickle
import warnings
import argparse
import numpy as np
import networkx as nx
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from copy import deepcopy


warnings.filterwarnings('ignore')
np.set_printoptions(suppress=True, linewidth=120, precision=2)

# np.random.seed(42)  # 3016222231 76(54)

current_seed = np.random.get_state()[1][0]

with open('granary.txt', 'a+') as o:
    o.write(str(current_seed)+'\n')


def args_parser(args):
    name = args.Name
    run = args.Run
    rho = args.Rho
    xi = args.Xi
    with_noise = args.With_noise
    pp = args.PreProcessing
    setting = args.Setting
    n_clusters = args.N_clusters
    max_iterations = args.Max_iterations

    return name, run, rho, xi, with_noise, pp, setting, n_clusters, max_iterations


def preprocess_y(y_in, data_type):
    """
    input:
    - Y: numpy array for Entity-to-feature
    - nscf: Dict. the dict-key is the index of categorical variable V_l in Y, and dict-value is the number of
    sub-categorie b_v (|V_l|) in categorical feature V_l.

    App_ly Z-scoring, preprocessing by range, and Prof. Mirkin's 3-stage pre-processing methods.
    return: Original entity-to-feature data matrix, Z-scored preprocessed matrix, 2-stages preprocessed matrix,
    3-stages preprocessed matrix and their coresponding relative contribution
    """

    TY = np.sum(np.multiply(y_in, y_in))  # data scatter, the letter T stands for data scatter
    TY_v = np.sum(np.multiply(y_in, y_in), axis=0)  # feature scatter
    Y_rel_cntr = TY_v / TY  # relative contribution

    Y_mean = np.mean(y_in, axis=0)
    Y_std = np.std(y_in, axis=0)

    y_z = np.divide(np.subtract(y_in, Y_mean), Y_std)  # Z-score

    Ty_z = np.sum(np.multiply(y_z, y_z))
    Ty_z_v = np.sum(np.multiply(y_z, y_z), axis=0)
    y_z_rel_cntr = Ty_z_v / Ty_z

    # scale_min_Y = np.min(y_in, axis=0)
    # scale_max_Y = np.max(y_in, axis=0)
    # rng_Y = scale_max_Y - scale_min_Y
    rng_Y = np.ptp(y_in, axis=0)

    y_rng = np.divide(np.subtract(y_in, Y_mean), rng_Y)  # 3 steps pre-processing (Range-without follow-up division)
    Ty_rng = np.sum(np.multiply(y_rng, y_rng))
    Ty_rng_v = np.sum(np.multiply(y_rng, y_rng), axis=0)
    y_rng_rel_cntr = Ty_rng_v / Ty_rng

    # This section is not used for sy_nthetic data, because no categorical data is generated.
    y_rng_rs = deepcopy(y_rng)  # 3 steps preprocessing (Range-with follow-up division)

    nscf = {}
    if data_type.lower() == 'c':
        for i in range(y_in.shape[1]):
            nscf[str(i)] = len(set(y_in[:, i]))

        col_counter = 0
        for k, v in nscf.items():
            col_counter += v
            if int(k) == 0:
                y_rng_rs[:, 0: col_counter] = y_rng_rs[:, 0: col_counter] / np.sqrt(int(v))
            if 0 < int(k) < y_in.shape[1]:
                y_rng_rs[:, col_counter: col_counter + v] = y_rng_rs[:, col_counter: col_counter + v] / np.sqrt(int(v))
            if int(k) == y_in.shape[1]:
                y_rng_rs[:, col_counter:] = y_rng_rs[:, col_counter:] / np.sqrt(int(v))

    # y_rng_rs = (Y_rescale - Y_mean)/ rng_Y
    Ty_rng_rs = np.sum(np.multiply(y_rng_rs, y_rng_rs))
    Ty_rng_v_rs = np.sum(np.multiply(y_rng_rs, y_rng_rs), axis=0)
    y_rng_rel_cntr_rs = Ty_rng_v_rs / Ty_rng_rs

    return y_in, Y_rel_cntr, y_z, y_z_rel_cntr, y_rng_rs, y_rng_rel_cntr_rs  # y_rng, y_rng_rel_cntr


def preprocess_p(p):

    """
    input: Adjacency matrix
    App_ly Uniform, Modularity, Lapin preprocessing methods.
    return: Original Adjanceny matrix, Uniform preprocessed matrix, Modularity preprocessed matrix, and
    Lapin preprocessed matrix and their coresponding relative contribution
    """
    N, V = p.shape
    p_sum_sim = np.sum(p)
    p_ave_sim = np.sum(p) / N * (V - 1)
    cnt_rnd_interact = np.mean(p, axis=1)  # constant random interaction

    # Uniform method
    p_u = p - cnt_rnd_interact
    p_u_sum_sim = np.sum(p_u)
    p_u_ave_sim = np.sum(p_u) / N * (V - 1)

    # Modularity method (random interaction)
    p_row = np.sum(p, axis=0)
    p_col = np.sum(p, axis=1)
    p_tot = np.sum(p)
    rnd_interact = np.multiply(p_row, p_col) / p_tot  # random interaction formula
    p_m = p - rnd_interact
    p_m_sum_sim = np.sum(p_m)
    p_m_ave_sim = np.sum(p_m) / N * (V - 1)

    # Lapin (Lap_lacian Inverse Transform)
    # Lap_lacian
    """
    r, c = P.shape
    P = (P + P.T) / 2  # to warrant the symmetry
    Pr = np.sum(P, axis=1)
    D = np.diag(Pr)
    D = np.sqrt(D)
    Di = LA.p_inv(D)
    L = eye(r) - Di @ P @ Di

    # pseudo-inverse transformation
    L = (L + L.T) / 2
    M, Z = LA.eig(L)  # eig-val, eig-vect
    ee = np.diag(M)
    print("ee:", ee)
    ind = list(np.nonzero(ee > 0)[0])  # indices of non-zero eigenvalues
    Zn = Z[ind, ind]
    print("Z:", Z)
    print("M:")
    print(M)
    print("ind:", ind)
    Mn = np.diag(M[ind])  # previously: Mn =  np.asarray(M[ind])
    print("Mn:", Mn)
    Mi = LA.inv(Mn)
    p_l = Zn@Mi@Zn.T
    """
    g = nx.from_numpy_array(p)
    g_p_l = nx.laplacian_matrix(g)
    p_l = np.asarray(g_p_l.todense())
    p_l_sum_sim = np.sum(p_l)
    p_l_ave_sim = np.sum(p_l) / N * (V - 1)

    return p, p_sum_sim, p_ave_sim, p_u, p_u_sum_sim, p_u_ave_sim, p_m, p_m_sum_sim, \
           p_m_ave_sim, p_l, p_l_sum_sim, p_l_ave_sim


def flat_cluster_results(cluster_results, version_new=True):
    
    if version_new is False:
        N = len([item for sublist in cluster_results.values() for item in sublist])
        labels_pred, labels_pred_indices = np.zeros([N]), []
        k = 1
        for key, v in cluster_results.items():
            for vv in v:
                labels_pred[vv] = k
                labels_pred_indices.append(vv)
            k += 1
    else:
        labels_pred = list(cluster_results.values())
        labels_pred_indices = list(cluster_results.key())

    return labels_pred, labels_pred_indices


def flat_ground_truth(ground_truth):
    """
    :param ground_truth: the clusters/communities cardinality
                        (output of cluster cardinality from synthetic data generator)
    :return: two flat lists, the first one is the list of labels in an appropriate format
             for applying sklearn metrics. And the second list is the list of lists of
              containing indices of nodes in the corresponding cluster.
    """
    k = 1
    interval = 1
    labels_true, labels_true_indices = [], []
    for v in ground_truth:
        tmp_indices = []
        for vv in range(v):
            labels_true.append(k)
            tmp_indices.append(interval+vv)

        k += 1
        interval += v
        labels_true_indices += tmp_indices

    return labels_true, labels_true_indices


def select_neighbours(p, node):

    """
    :param p: N*N adjacency matrix
    :param node: index of a node
    :return: a shuffled list of neighbour nodes
    """
    neighbors = np.where(p[node, :] > 0)[0].tolist()
    neighbors = random.sample(neighbors, len(neighbors))
    # neighbors = list(range(p.shape[0]))
    # print("neighbors:", neighbors)
    return neighbors


def compute_gain(y, p, rho, xi, indices):

    cv_y = np.sum(np.power(np.mean(y[indices, :], axis=0), 2))  # Cv in the criterion
    ss_y = np.sum([p[i, j] for i in indices for j in indices])  # summary similarities
    Lambda = np.divide(ss_y, len(indices) ** 2)
    gain = rho * cv_y * len(indices) + xi * Lambda * ss_y

    return gain


def phase_one(y, y_original, p, p_original, labels, rho, xi, greedy):

    """
    :param y: N*V entity-to-feature matrix
    :param p: N*N adjacency matrix
    :param labels: dict where keys represents the nodes index and values represents the corresponding community label
    :param rho:
    :param xi:
    :param greedy:
    :return: list of sets, each list represents the indices of nodes of a cluster (updated version of input labels)
    """

    clusters_labels = list(set(labels.values()))
    clusters_number = len(clusters_labels)
    clusters_indices = {label: [] for label in clusters_labels}

    # grouping clusters w.r.t their indices
    for k, v in labels.items():
        for label in clusters_labels:
            if v == label:
                clusters_indices[label].append(k)

    # print("clusters_indices:", clusters_indices)

    for node in range(y.shape[0]):
        label = labels[node]

        # print("node:", node, "current community:", label, "n_labels:", len(set(labels.values())))
        # print('old labels:', labels)

        tmp_indices = [k for k, v in labels.items() if v == label]

        gain_old = compute_gain(y=y_original, p=p_original, rho=rho, xi=xi, indices=tmp_indices)
        # print("old gain:", gain_old)
        neighbours = select_neighbours(p=p, node=node)
        # print("neighbours:", neighbours)

        if greedy is True:

            tmp_gains = np.zeros([y.shape[0]])

        for neighbour in neighbours:

            tmp_indices = [k for k, v in labels.items() if v == label]
            tmp_indices += [neighbour]

            gain_new = compute_gain(y=y_original, p=p_original, rho=rho, xi=xi, indices=tmp_indices)

            # print("new gain:", gain_new)
            delta = gain_new - gain_old

            if greedy is True and delta > 0:
                tmp_gains[neighbour] = delta
            elif greedy is False and delta > 0:
                # labels[node] = labels[neighbour]
                for i in clusters_indices[label]:
                    labels[i] = labels[neighbour]
                break
        if greedy is True:
            # labels[node] = labels[np.argmax(tmp_gains)]
            for i in clusters_indices[label]:
                labels[i] = labels[np.argmax(tmp_gains)]

            # print("new labels for node:", node, "arg max", np.argmax(tmp_gains),
            #       labels[np.argmax(tmp_gains)], "n_labels:", len(set(labels.values())))

            # print("new labels:", labels)

    print("end of 1st stage!")

    return labels


def phase_two(y_original, p_original, labels, directed=False):

    clusters_labels = list(set(labels.values()))
    clusters_number = len(clusters_labels)
    clusters_indices = {label: [] for label in range(clusters_number)}

    clusters_mean = {}
    updated_labels = {}
    within_cluster_summary_similarities = {}
    between_clusters_summary_similarities = {}

    # grouping clusters w.r.t their indices
    for k, v in labels.items():
        for label in range(len(clusters_labels)):
            if v == clusters_labels[label]:
                clusters_indices[label].append(k)
                updated_labels[k] = label

    # computing the average and summary similarity for moving the community centers
    for label, wth_indices in clusters_indices.items():  # wth_indices >> within cluster indices
        # print("wth_indices:", wth_indices,)
        clusters_mean[label] = np.mean(y_original[wth_indices, :], axis=0)
        within_cluster_summary_similarities[label] = np.sum([p_original[i, j] for i in
                                                             wth_indices for j in wth_indices])

    # computing the summary similarity between clusters
    for label, wth_indices in clusters_indices.items():
        for label_, wth_indices_ in clusters_indices.items():

            if label != label_:

                if directed is False:
                    between_clusters_summary_similarities[str(label) + "-" + str(label_)] = 2 * np.sum(
                        [p_original[i, j] for i in wth_indices for j in wth_indices_])
                else:
                    first_direction = np.sum([p_original[i, j] for i in wth_indices for j in wth_indices_])
                    second_direction = np.sum([p_original[i, j] for i in wth_indices_ for j in wth_indices])
                    between_clusters_summary_similarities[str(label) + "-" + str(label_)] = (first_direction,
                                                                                             second_direction)

    # print("clusters_mean:", clusters_mean)
    # print("within_cluster_summary_similarities:", within_cluster_summary_similarities)
    # print("between_clusters_summary_similarities:", between_clusters_summary_similarities)

    # merging the two clusters

    # clusters_number = clusters_number - 1
    y_agg = np.zeros([clusters_number, y_original.shape[1]])
    p_agg = np.zeros([clusters_number, clusters_number])

    # Aggregation of features part and within cluster summary similarities
    for k, v in clusters_mean.items():
        # print("k", k, "c:", v, "s:",  within_cluster_summary_similarities[k])
        y_agg[k, :] = v
        p_agg[k, k] = within_cluster_summary_similarities[k]

    # Aggregation of features part and between cluster summary similarities
    for k, v in between_clusters_summary_similarities.items():
        if directed is False:
            idx = k.split("-")
            p_agg[int(idx[0]), int(idx[1])] = v
            p_agg[int(idx[1]), int(idx[0])] = v
        else:
            idx = k.split("-")
            p_agg[int(idx[0]), int(idx[1])] = v[0]
            p_agg[int(idx[1]), int(idx[0])] = v[1]

    print("aggregate matrices:", y_agg.shape, p_agg.shape,)
    # print(y_agg)
    # print(p_agg)

    return y_agg, p_agg, updated_labels


def run_louvain(y, p, labels, rho, xi, n_clusters, max_iterations):

    f1 = True

    y_original = deepcopy(y)
    p_original = deepcopy(p)

    print("y_org:")
    print(y_original)

    print("p_org:")
    print(p_original)

    while f1 is True:

        labels_old = deepcopy(labels)

        labels_new = phase_one(y=y, y_original=y_original, p=p, p_original=p_original,
                               labels=labels, rho=rho, xi=xi, greedy=True)

        # print("labels old:", len(set(labels_old.values())), labels_old, )
        # print(" ")
        # print("labels after 1st phase:", len(set(labels_new.values())), labels_new)
        # print(" ")

        cntr = 0
        for k, v in labels_new.items():
            if v == labels_old[k]:
                cntr += 1
            elif v != labels_old[k]:
                cntr = 0
                break

        labels = deepcopy(labels_new)

        # End of phase one, that is entity swing does not improve the gain anymore
        # if list(labels.values()) == list(labels_old.values()):

        print("Applying phase two by merging nodes of the same community and repeating the 1st phase")

        y_agg, p_agg, updated_labels = phase_two(y_original=y_original, p_original=p_original,
                                                 labels=labels, directed=False)
        y = deepcopy(y_agg)
        p = deepcopy(p_agg)
        labels = deepcopy(updated_labels)

        n_detected_clusters = len(set(labels.values()))
        print("n_clusters:", n_clusters, "n_detected_clusters:", n_detected_clusters,
              "max_iterations:", max_iterations)
        max_iterations -= 1

        if n_detected_clusters <= n_clusters or max_iterations == 0:
            f1 = False
            updated_labels = {}
            clusters_indices = []
            print("Finishing")
            label_pred = 0
            clusters_labels = list(set(labels.values()))
            for k, v in labels.items():
                for label in range(len(clusters_labels)):
                    if v == clusters_labels[label]:
                        clusters_indices.append(k)
                        updated_labels[k] = label

            labels = deepcopy(updated_labels)
            print("labels final:", labels)

    return labels


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--Name', type=str, default='--',
                        help='Name of the Experiment')

    parser.add_argument('--Run', type=int, default=0,
                        help='Whether to run the program or to evaluate the results')

    parser.add_argument('--Rho', type=float, default=1,
                        help='Feature coefficient during the clustering')

    parser.add_argument('--Xi', type=float, default=1,
                        help='Networks coefficient during the clustering')

    parser.add_argument('--With_noise', type=int, default=0,
                        help='With noisy features or without')

    parser.add_argument('--PreProcessing', type=str, default='z-m',
                        help='string determining which pre processing method should be app_lied.'
                             'The first letter determines Y pre processing and the third determines P pre processing. '
                             'Seperated with "-".')

    parser.add_argument('--Setting', type=str, default='all')

    parser.add_argument('--N_clusters', type=int, default=5,
                        help='Determine the number of detected clusters')

    parser.add_argument('--Max_iterations', type=int, default=50,
                        help='Maximum number of iterations for optimizing the criterion in a local search')

    args = parser.parse_args()
    name, run, rho, xi, with_noise, pp, setting_, n_clusters, max_iterations = args_parser(args)

    data_name = name.split('(')[0]
    if with_noise == 1:
        data_name = data_name + "-N"
    type_of_data = name.split('(')[0][-1]
    print("run:", run, name, pp, with_noise, setting_, data_name, type_of_data)

    start = time.time()
    
    if run == 1:

        with open(os.path.join('../data', name + ".pickle"), 'rb') as fp:
            DATA = pickle.load(fp)

        print("run:", run, rho, xi, name, pp, with_noise, setting_)

        def apply_lbcd_fnr(data_type, with_noise):

            # Global initialization
            out_ms = {}

            if setting_ != 'all':
                for setting, repeats in DATA.items():

                    if str(setting) == setting_:

                        print("setting:", setting, )

                        out_ms[setting] = {}
                        
                        for repeat, matrices in repeats.items():
                            print("repeat:", repeat)
                            gt = matrices['GT']
                            y = matrices['Y']
                            p = matrices['P']
                            y_n = matrices['Yn']
                            n, v = y.shape
                            labels = {k: k for k in range(y.shape[0])}

                            # Quantitative case
                            if type_of_data == 'Q' or name.split('(')[-1] == 'r':
                                _, _, y_z, _, y_rng, _, = preprocess_y(y_in=y, data_type='Q')

                                if with_noise == 1:
                                    y_n, _, y_n_z, _, y_n_rng, _, = preprocess_y(y_in=y_n, data_type='Q')

                            # Because there is no y_n in the case of categorical features.
                            if type_of_data == 'C':
                                enc = OneHotEncoder(sparse=False, )  # categories='auto')
                                y_onehot = enc.fit_transform(y)  # oneHot encoding

                                # for WITHOUT follow-up rescale y_onehot and for
                                # "WITH follow-up" y_onehot should be rep_laced with Y
                                y, _, y_z, _, y_rng, _, = preprocess_y(y_in=y_onehot, data_type='C')  # y_onehot

                            if type_of_data == 'M':
                                v_q = int(np.ceil(v/2))  # number of quantitative features -- Y[:, :v_q]
                                v_c = int(np.floor(v/2))  # number of categorical features  -- Y[:, v_q:]
                                _, _, y_z_q, _, y_rng_q, _, = preprocess_y(y_in=y[:, :v_q], data_type='Q')
                                enc = OneHotEncoder(sparse=False, )  # categories='auto', )
                                y_onehot = enc.fit_transform(y[:, v_q:])  # oneHot encoding

                                # for WITHOUT follow-up rescale y_onehot and for
                                # "WITH follow-up" y_onehot should be rep_laced with Y
                                _, _, y_z_c, _, y_rng_c, _, = preprocess_y(y_in=y[:, v_q:], data_type='C')  # y_onehot
                                y = np.concatenate([y[:, :v_q], y_onehot], axis=1)
                                y_rng = np.concatenate([y_rng_q, y_rng_c], axis=1)
                                y_z = np.concatenate([y_z_q, y_z_c], axis=1)

                                if with_noise == 1:
                                    v_q = int(np.ceil(v/2))  # number of quantitative features -- Y[:, :v_q]
                                    v_c = int(np.floor(v/2))  # number of categorical features  -- Y[:, v_q:]
                                    v_qn = (v_q + v_c)  # the column index of which noise model1 starts

                                    _, _, y_n_z_q, _, y_n_rng_q, _, = preprocess_y(y_in=y_n[:, :v_q], data_type='Q')

                                    enc = OneHotEncoder(sparse=False, )  # categories='auto',)
                                    y_n_onehot = enc.fit_transform(y_n[:, v_q:v_qn])  # oneHot encoding

                                    # for WITHOUT follow-up rescale y_n_oneHot and for
                                    # "WITH follow-up" y_n_oneHot should be rep_laced with Y
                                    y_n_c, _, y_n_z_c, _, y_n_rng_c, _, = preprocess_y(y_in=y_n[:, v_q:v_qn],
                                                                                  data_type='C')  # y_n_oneHot

                                    y_ = np.concatenate([y_n[:, :v_q], y_n_onehot], axis=1)
                                    y_rng = np.concatenate([y_n_rng_q, y_n_rng_c], axis=1)
                                    y_z = np.concatenate([y_n_z_q, y_n_z_c], axis=1)

                                    _, _, y_n_z_, _, y_n_rng_, _, = preprocess_y(y_in=y_n[:, v_qn:], data_type='Q')
                                    y_n_ = np.concatenate([y_, y_n[:, v_qn:]], axis=1)
                                    y_n_rng = np.concatenate([y_rng, y_n_rng_], axis=1)
                                    y_n_z = np.concatenate([y_z, y_n_z_], axis=1)

                            p, _, _, p_u, _, _, p_m, _, _, p_l, _, _ = preprocess_p(p=p)

                            # Pre-processing - Without Noise
                            if data_type == "NP".lower() and with_noise == 0:
                                print("NP")
                                tmp_ms = run_louvain(y=y, p=p, labels=labels,
                                                     rho=rho, xi=xi, n_clusters=n_clusters,
                                                     max_iterations=max_iterations)

                            elif data_type == "z-n".lower() and with_noise == 0:
                                print("z-n")
                                tmp_ms = run_louvain(y=y_z, p=p, labels=labels,
                                                     rho=rho, xi=xi, n_clusters=n_clusters,
                                                     max_iterations=max_iterations)

                            elif data_type == "rng-n".lower() and with_noise == 0:
                                print("z-n")
                                tmp_ms = run_louvain(y=y_rng, p=p, labels=labels,
                                                     rho=rho, xi=xi, n_clusters=n_clusters,
                                                     max_iterations=max_iterations)

                            elif data_type == "z-u".lower() and with_noise == 0:
                                print("z-u")
                                tmp_ms = run_louvain(y=y_z, p=p_u, labels=labels,
                                                     rho=rho, xi=xi, n_clusters=n_clusters,
                                                     max_iterations=max_iterations)

                            elif data_type == "z-m".lower() and with_noise == 0:
                                tmp_ms = run_louvain(y=y_z, p=p_m, labels=labels,
                                                     rho=rho, xi=xi, n_clusters=n_clusters,
                                                     max_iterations=max_iterations)

                            elif data_type == "z-l".lower() and with_noise == 0:
                                tmp_ms = run_louvain(y=y_z, p=p_l, labels=labels,
                                                     rho=rho, xi=xi, n_clusters=n_clusters,
                                                     max_iterations=max_iterations)

                            elif data_type == "rng-u".lower() and with_noise == 0:
                                tmp_ms = run_louvain(y=y_rng, p=p_u, labels=labels,
                                                     rho=rho, xi=xi, n_clusters=n_clusters,
                                                     max_iterations=max_iterations)

                            elif data_type == "rng-m".lower() and with_noise == 0:
                                tmp_ms = run_louvain(y=y_rng, p=p_m, labels=labels,
                                                     rho=rho, xi=xi, n_clusters=n_clusters,
                                                     max_iterations=max_iterations)

                            elif data_type == "rng-l".lower() and with_noise == 0:
                                tmp_ms = run_louvain(y=y_rng, p=p_l, labels=labels,
                                                     rho=rho, xi=xi, n_clusters=n_clusters,
                                                     max_iterations=max_iterations)

                            # Pre-processing - With Noise
                            if data_type == "NP".lower() and with_noise == 1:
                                tmp_ms = run_louvain(y=y_n, p=p, labels=labels, 
                                                     rho=rho, xi=xi, n_clusters=n_clusters,
                                                     max_iterations=max_iterations)

                            elif data_type == "z-u".lower() and with_noise == 1:
                                tmp_ms = run_louvain(y=y_n_z, p=p_u, labels=labels, 
                                                     rho=rho, xi=xi, n_clusters=n_clusters,
                                                     max_iterations=max_iterations)

                            elif data_type == "z-m".lower() and with_noise == 1:
                                tmp_ms = run_louvain(y=y_n_z, p=p_m, labels=labels,
                                                     rho=rho, xi=xi, n_clusters=n_clusters,
                                                     max_iterations=max_iterations)

                            elif data_type == "z-l".lower() and with_noise == 1:
                                tmp_ms = run_louvain(y=y_n_z, p=p_l, labels=labels,
                                                     rho=rho, xi=xi, n_clusters=n_clusters,
                                                     max_iterations=max_iterations)

                            elif data_type == "rng-u".lower() and with_noise == 1:
                                tmp_ms = run_louvain(y=y_n_rng, p=p_u, labels=labels, 
                                                    rho=rho, xi=xi, n_clusters=n_clusters,
                                                    max_iterations=max_iterations)

                            elif data_type == "rng-m".lower() and with_noise == 1:
                                tmp_ms = run_louvain(y=y_n_rng, p=p_m, labels=labels,
                                                     rho=rho, xi=xi, n_clusters=n_clusters,
                                                     max_iterations=max_iterations)

                            elif data_type == "rng-l".lower() and with_noise == 1:
                                tmp_ms = run_louvain(y=y_n_rng, p=p_l, labels=labels,
                                                     rho=rho, xi=xi, n_clusters=n_clusters,
                                                     max_iterations=max_iterations)

                            out_ms[setting][repeat] = tmp_ms

                    print("Algorithm is app_lied on the entire data set!")

            if setting_ == 'all':

                for setting, repeats in DATA.items():

                    print("setting:", setting, )

                    out_ms[setting] = {}
                    for repeat, matrices in repeats.items():
                        print("repeat:", repeat)
                        gt = matrices['GT']
                        y = matrices['Y']
                        p = matrices['P']
                        y_n = matrices['Yn']
                        n, v = y.shape
                        labels = {k: k for k in range(y.shape[0])}

                        # Quantitative case
                        if type_of_data == 'Q' or name.split('(')[-1] == 'r':
                            _, _, y_z, _, y_rng, _, = preprocess_y(y_in=y, data_type='Q')

                            if with_noise == 1:
                                y_n, _, y_n_z, _, y_n_rng, _, = preprocess_y(y_in=y_n, data_type='Q')

                        # Because there is no y_n in the case of categorical features.
                        if type_of_data == 'C':
                            enc = OneHotEncoder()  # categories='auto')
                            y = enc.fit_transform(y)  # oneHot encoding
                            y = y.toarray()
                            # Boris's Theory
                            y, _, y_z, _, y_rng, _, = preprocess_y(y_in=y, data_type='C')

                        if type_of_data == 'M':
                            v_q = int(np.ceil(v/2))  # number of quantitative features -- Y[:, :v_q]
                            v_c = int(np.floor(v/2))  # number of categorical features  -- Y[:, v_q:]
                            Y_, _, y_z_, _, y_rng_, _, = preprocess_y(y_in=Y[:, :v_q], data_type='M')
                            enc = OneHotEncoder(sparse=False, )  # categories='auto', )
                            y_onehot = enc.fit_transform(Y[:, v_q:])  # oneHot encoding
                            Y = np.concatenate([y_onehot, Y[:, :v_q]], axis=1)
                            y_rng = np.concatenate([y_onehot, y_rng_], axis=1)
                            y_z = np.concatenate([y_onehot, y_z_], axis=1)

                            if with_noise == 1:
                                v_q = int(np.ceil(v/2))  # number of quantitative features -- Y[:, :v_q]
                                v_c = int(np.floor(v/2))  # number of categorical features  -- Y[:, v_q:]
                                v_qn = (v_q + v_c)  # the column index of which noise model1 starts

                                _, _, y_z_, _, y_rng_, _, = preprocess_y(y_in=y_n[:, :v_q], data_type='M')
                                enc = OneHotEncoder(sparse=False, )  # categories='auto',)
                                y_n_oneHot = enc.fit_transform(y_n[:, v_q:v_qn])  # oneHot encoding
                                Y_ = np.concatenate([y_n_oneHot, y_n[:, :v_q]], axis=1)
                                y_rng = np.concatenate([y_n_oneHot, y_rng_], axis=1)
                                y_z = np.concatenate([y_n_oneHot, y_z_], axis=1)

                                _, _, y_n_z_, _, y_n_rng_, _, = preprocess_y(y_in=y_n[:, v_qn:], data_type='M')
                                y_n_ = np.concatenate([Y_, y_n[:, v_qn:]], axis=1)
                                y_n_rng = np.concatenate([y_rng, y_n_rng_], axis=1)
                                y_n_z = np.concatenate([y_z, y_n_z_], axis=1)

                        p, _, _, p_u, _, _, p_m, _, _, p_l, _, _ = preprocess_p(p=p)

                        # Pre-processing - Without Noise
                        if data_type == "NP".lower() and with_noise == 0:
                            tmp_ms = run_louvain(y=y, p=p, labels=labels,
                                                 rho=rho, xi=xi, n_clusters=n_clusters,
                                                 max_iterations=max_iterations)

                        elif data_type == "z-n".lower() and with_noise == 0:
                            print("z-n")
                            tmp_ms = run_louvain(y=y_z, p=p, labels=labels,
                                                 rho=rho, xi=xi, n_clusters=n_clusters,
                                                 max_iterations=max_iterations)

                        elif data_type == "rng-n".lower() and with_noise == 0:
                            print("z-n")
                            tmp_ms = run_louvain(y=y_rng, p=p, labels=labels,
                                                 rho=rho, xi=xi, n_clusters=n_clusters,
                                                 max_iterations=max_iterations)

                        elif data_type == "z-u".lower() and with_noise == 0:
                            tmp_ms = run_louvain(y=y_z, p=p_u, labels=labels,
                                                 rho=rho, xi=xi, n_clusters=n_clusters,
                                                 max_iterations=max_iterations)

                        elif data_type == "z-m".lower() and with_noise == 0:
                            tmp_ms = run_louvain(y=y_z, p=p_m, labels=labels,
                                                 rho=rho, xi=xi, n_clusters=n_clusters,
                                                 max_iterations=max_iterations)

                        elif data_type == "z-l".lower() and with_noise == 0:
                            tmp_ms = run_louvain(y=y_z, p=p_l, labels=labels,
                                                 rho=rho, xi=xi, n_clusters=n_clusters,
                                                 max_iterations=max_iterations)

                        elif data_type == "rng-u".lower() and with_noise == 0:
                            tmp_ms = run_louvain(y=y_rng, p=p_u, labels=labels,
                                                 rho=rho, xi=xi, n_clusters=n_clusters,
                                                 max_iterations=max_iterations)

                        elif data_type == "rng-m".lower() and with_noise == 0:
                            tmp_ms = run_louvain(y=y_rng, p=p_m, labels=labels,
                                                 rho=rho, xi=xi, n_clusters=n_clusters,
                                                 max_iterations=max_iterations)

                        elif data_type == "rng-l".lower() and with_noise == 0:
                            tmp_ms = run_louvain(y=y_rng, p=p_l, labels=labels,
                                                 rho=rho, xi=xi, n_clusters=n_clusters,
                                                 max_iterations=max_iterations)

                        # Pre-processing - With Noise
                        if data_type == "NP".lower() and with_noise == 1:
                            tmp_ms = run_louvain(y=y_n, p=p, labels=labels,
                                                 rho=rho, xi=xi, n_clusters=n_clusters,
                                                 max_iterations=max_iterations)

                        elif data_type == "z-u".lower() and with_noise == 1:
                            tmp_ms = run_louvain(y=y_n_z, p=p_u, labels=labels,
                                                 rho=rho, xi=xi, n_clusters=n_clusters,
                                                 max_iterations=max_iterations)

                        elif data_type == "z-m".lower() and with_noise == 1:
                            tmp_ms = run_louvain(y=y_n_z, p=p_m, labels=labels,
                                                 rho=rho, xi=xi, n_clusters=n_clusters,
                                                 max_iterations=max_iterations)

                        elif data_type == "z-l".lower() and with_noise == 1:
                            tmp_ms = run_louvain(y=y_n_z, p=p_l, labels=labels,
                                                 rho=rho, xi=xi, n_clusters=n_clusters,
                                                 max_iterations=max_iterations)

                        elif data_type == "rng-u".lower() and with_noise == 1:
                            tmp_ms =run_louvain(y=y_n_rng, p=p_u, labels=labels,
                                                rho=rho, xi=xi, n_clusters=n_clusters,
                                                max_iterations=max_iterations)

                        elif data_type == "rng-m".lower() and with_noise == 1:
                            tmp_ms = run_louvain(y=y_n_rng, p=p_m, labels=labels,
                                                 rho=rho, xi=xi, n_clusters=n_clusters,
                                                 max_iterations=max_iterations)

                        elif data_type == "rng-l".lower() and with_noise == 1:
                            tmp_ms = run_louvain(y=y_n_rng, p=p_l, labels=labels,
                                                 rho=rho, xi=xi, n_clusters=n_clusters,
                                                 max_iterations=max_iterations)

                        out_ms[setting][repeat] = tmp_ms

                print("Algorithm is app_lied on the entire data set!")

            return out_ms


        out_ms = apply_lbcd_fnr(data_type=pp.lower(), with_noise=with_noise)

        end = time.time()
        print("Time:", end - start)

        if with_noise == 1:
            name = name + '-N'

        if setting_ != 'all':
            with open(os.path.join('../data', "out_ms_" + name + "-" + pp + "-" + setting_ + ".pickle"),
                      'wb') as fp:
                pickle.dump(out_ms, fp)

        if setting_ == 'all':
            with open(os.path.join('../data', "out_ms_" + name + "-" + pp + "-" + ".pickle"), 'wb') as fp:
                pickle.dump(out_ms, fp)

        print("Results are saved!")

        print(" ")
        print("\t", "  p", "  q", " a/e", "\t", "  ARI     ", "  NMI", )
        print(" \t", " \t", " \t", " Ave", " std", " Ave", " std")
        for setting, results in out_ms.items():
            ARI, NMI = [], []
            for repeat, result in results.items():
                lp, lpi = list(result.values()), list(result.keys())
                if not name.split('(')[-1] == 'r':
                    gt, gti = flat_ground_truth(DATA[setting][repeat]['GT'])
                else:
                    gt = DATA[setting][repeat]['GT']

                ARI.append(metrics.adjusted_rand_score(gt, lp))
                NMI.append(metrics.adjusted_mutual_info_score(gt, lp))

            ari_ave = np.mean(np.asarray(ARI), axis=0)
            ari_std = np.std(np.asarray(ARI), axis=0)
            nmi_ave = np.mean(np.asarray(NMI), axis=0)
            nmi_std = np.std(np.asarray(NMI), axis=0)
            print("setting:", setting, "%.2f" % ari_ave, "%.2f" % ari_std, "%.2f" % nmi_ave,
                  "%.2f" % nmi_std)

        print(" ")
        print("\t", "  p", "  q", " a/e   ", "precision,", 'recall', '  f-score')
        print(" \t", " \t", " \t", "Ave", " std", " Ave", " std", " Ave", " std")
        for setting, results in out_ms.items():
            precision, recall, fscore = [], [], []
            for repeat, result in results.items():
                lp, lpi = list(result.values()), list(result.keys())

                if not name.split('(')[-1] == 'r':
                    gt, gti = flat_ground_truth(DATA[setting][repeat]['GT'])
                else:
                    gt = DATA[setting][repeat]['GT']

                tmp = metrics.precision_recall_fscore_support(gt, lp, average='weighted')

                precision.append(tmp[0])
                recall.append(tmp[1])
                fscore.append(tmp[2])

            precision_ave = np.mean(np.asarray(precision), axis=0)
            precision_std = np.std(np.asarray(precision), axis=0)

            recall_ave = np.mean(np.asarray(recall), axis=0)
            recall_std = np.std(np.asarray(recall), axis=0)

            fscore_ave = np.mean(np.asarray(fscore), axis=0)
            fscore_std = np.std(np.asarray(fscore), axis=0)

            print("setting:", setting, "%.2f" % precision_ave, "%.2f" % precision_std,
                  "%.2f" % recall_ave, "%.2f" % recall_std,
                  "%.2f" % fscore_ave, "%.2f" % fscore_std,
                  )

        print(" ")
        print(" Number of detected clusters")
        print(" \t", " \t", "   Ave", "  std", )
        for setting, results in out_ms.items():
            num_cluster = []
            for repeat, result in results.items():
                num_cluster.append(int(len(set(result.values()))))

            ave_num_clust = np.mean(np.asarray(num_cluster), axis=0)
            std_num_clust = np.std(np.asarray(num_cluster), axis=0)
            print("Number of Clusters:", ave_num_clust, std_num_clust)

        # for setting, results in out_ms.items():
        #     ARI, NMI = [], []
        #     for repeat, result in results.items():
        #         tmp_len_ms, tmp_out_ms = sorting_results(result)
        #         tmp_len_gt, tmp_out_gt = sorting_results(DATA[setting][repeat]['GT'])
        #         gt, _ = flat_ground_truth(tmp_out_gt)
        #         lp, _ = flat_ground_truth(tmp_out_ms)
        #         ARI.append(metrics.adjusted_rand_score(gt, lp))
        #         NMI.append(metrics.adjusted_mutual_info_score(gt, lp))
        #
        #     ari_ave_new = np.mean(np.asarray(ARI), axis=0)
        #     ari_std_new = np.std(np.asarray(ARI), axis=0)
        #     nmi_ave_new = np.mean(np.asarray(NMI), axis=0)
        #     nmi_std_new = np.std(np.asarray(NMI), axis=0)
        #     print("setting:", setting, "%.2f" % ari_ave_new, "%.2f" % ari_std_new, "%.2f" % nmi_ave_new,
        #           "%.2f" % nmi_std_new)

    if run == 0:

        print(" \t", " \t", "name:", name)

        with open(os.path.join('../data', name + ".pickle"), 'rb') as fp:
            DATA = pickle.load(fp)

        if with_noise == 1:
            name = name + '-N'
        with open(os.path.join('../data', "out_ms_" + name + "-" + pp + ".pickle"), 'rb') as fp:
            out_ms = pickle.load(fp)

        print(" ")
        print("\t", "  p", "  q", " a/e", "\t", "  ARI     ", "  NMI", )
        print(" \t", " \t", " \t", " Ave", " std", " Ave", " std")
        for setting, results in out_ms.items():
            ARI, NMI = [], []
            for repeat, result in results.items():
                lp, lpi = list(result.values()), list(result.keys())

                if not name.split('(')[-1] == 'r':
                    gt, gti = flat_ground_truth(DATA[setting][repeat]['GT'])
                else:
                    gt = DATA[setting][repeat]['GT']

                ARI.append(metrics.adjusted_rand_score(gt, lp))
                NMI.append(metrics.adjusted_mutual_info_score(gt, lp))

            ari_ave = np.mean(np.asarray(ARI), axis=0)
            ari_std = np.std(np.asarray(ARI), axis=0)
            nmi_ave = np.mean(np.asarray(NMI), axis=0)
            nmi_std = np.std(np.asarray(NMI), axis=0)
            print("setting:", setting, "%.2f" % ari_ave, "%.2f" % ari_std, "%.2f" % nmi_ave,
                  "%.2f" % nmi_std)

        print(" ")
        print("\t", "  p", "  q", " a/e   ", "precision,", 'recall', '  f-score')
        print(" \t", " \t", " \t", "Ave", " std", " Ave", " std", " Ave", " std")
        for setting, results in out_ms.items():
            precision, recall, fscore = [], [], []
            for repeat, result in results.items():
                lp, lpi = list(result.values()), list(result.keys())

                if not name.split('(')[-1] == 'r':
                    gt, gti = flat_ground_truth(DATA[setting][repeat]['GT'])
                else:
                    gt = DATA[setting][repeat]['GT']

                tmp = metrics.precision_recall_fscore_support(gt, lp, average='weighted')

                precision.append(tmp[0])
                recall.append(tmp[1])
                fscore.append(tmp[2])

            precision_ave = np.mean(np.asarray(precision), axis=0)
            precision_std = np.std(np.asarray(precision), axis=0)

            recall_ave = np.mean(np.asarray(recall), axis=0)
            recall_std = np.std(np.asarray(recall), axis=0)

            fscore_ave = np.mean(np.asarray(fscore), axis=0)
            fscore_std = np.std(np.asarray(fscore), axis=0)

            print("setting:", setting, "%.2f" % precision_ave, "%.2f" % precision_std,
                  "%.2f" % recall_ave, "%.2f" % recall_std,
                  "%.2f" % fscore_ave, "%.2f" % fscore_std,
                  )

        print(" ")
        print(" Number of detected clusters")
        print(" \t", " \t", "   Ave", "  std", )
        for setting, results in out_ms.items():
            num_cluster = []
            for repeat, result in results.items():
                num_cluster.append(int(len(set(result.values()))))

            ave_num_clust = np.mean(np.asarray(num_cluster), axis=0)
            std_num_clust = np.std(np.asarray(num_cluster), axis=0)
            print("Number of Clusters:", "%.2f" % ave_num_clust, "%.2f" % std_num_clust)

        print("contingency tables")
        # for setting, results in out_ms.items():
        #     for repeat, result in results.items():
        #         lp, lpi = flat_cluster_results(result)
        #         gt, gti = flat_ground_truth(DATA[setting][repeat]['GT'])
        #         tmp_cont, _, _ = ev.sk_contingency(tmp_out_ms, tmp_out_gt,)  # N
        #         print("setting:", setting, repeat)
        #         print(tmp_cont)
        #         print(" ")

