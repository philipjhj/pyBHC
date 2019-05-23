from __future__ import print_function, division
import itertools as it
import numpy as np
import sys
from scipy.cluster.hierarchy import ClusterNode, dendrogram
from scipy import linalg
import matplotlib.pyplot as plt
import seaborn as sns
import logging

import matplotlib as mpl
from numpy import logaddexp
import math
from tqdm import tqdm
import pdb


logger = logging.getLogger(__name__)


class bhc(object):
    """
    An instance of Bayesian hierarchical clustering CRP mixture model.
    Attributes
    ----------
    assignments : list(list(int))
        A list of lists, where each list records the clustering at
        each step by giving the index of the leftmost member of the
        cluster a leaf is traced to.
    root_node : Node
        The root node of the clustering tree.
    lml : float
        An estimate of the log marginal likelihood of the model
        under a DPMM.
    Notes
    -----
    The cost of BHC scales as O(n^2) and so becomes inpractically
    large for datasets of more than a few hundred points.
    """

    def __init__(self, data, data_model, crp_alpha=1.0):
        """
        Init a bhc instance and perform the clustering.

        Parameters
        ----------
        data : numpy.ndarray (n, d)
            Array of data where each row is a data point and each
            column is a dimension.
        data_model : CollapsibleDistribution
            Provides the approprite ``log_marginal_likelihood``
            function for the data.
        crp_alpha : float (0, Inf)
            CRP concentration parameter.
        verbose : bool, optional
            Determibes whetrher info gets dumped to stdout.
        """
        self.data_model = data_model
        self.crp_alpha = crp_alpha

        # initialize the tree
        if not all(isinstance(n, Node) for n in data):
            self.nodes = [self.create_leaf_node(
                node_id, np.array([x])) for node_id, x in enumerate(data)]
        else:
            self.nodes = data

    def fit(self):
        assignment = list(range(len(self.nodes)))
        self.assignments = [list(assignment)]
        self.rks = []

        current_roots = assignment
        cached_nodes = dict()
        merged_node_id = max([node.id for node in self.nodes])
        while len(current_roots) > 1:
            max_rk = float('-Inf')
            merged_node = None

            # for each pair of clusters (nodes), compute the merger
            # score.

            logger.info('Computing next merge; missing {} merges'.format(
                len(current_roots)-1))

            # tqdm not integrated with logging
            # see https://stackoverflow.com/questions/14897756/python-progress-bar-through-logging-module/41224909
            for left_idx, right_idx in it.combinations(current_roots, 2):
                merged_node_id += 1
                if (left_idx, right_idx) not in cached_nodes.keys():
                    tmp_node = self.create_merged_node(merged_node_id,
                                                       self.nodes[left_idx],
                                                       self.nodes[right_idx])

                    cached_nodes[(left_idx, right_idx)] = tmp_node
                else:
                    tmp_node = cached_nodes[(left_idx, right_idx)]

                if tmp_node.log_rk > max_rk:
                    max_rk = tmp_node.log_rk
                    merged_node = tmp_node
                    merged_right = right_idx
                    merged_left = left_idx

            current_roots.remove(merged_left)
            current_roots.remove(merged_right)
            current_roots.append(len(self.nodes))

            # merged_node.log_rk = 0
            self.nodes.append(merged_node)

            del cached_nodes[(merged_left, merged_right)]

            self.rks.append(math.exp(max_rk))

            for i, k in enumerate(assignment):
                if k == merged_right:
                    assignment[i] = merged_left
            self.assignments.append(list(assignment))

        self.assignments = np.array(self.assignments)
        self.root_node = self.nodes[-1]

        self.omegas = self.compute_omegas(self.root_node)

    def randomized_fit(self, m_fraction=0.2):

        merged_node_id = max([node.id for node in self.nodes])
        self.rks = []

        def randomizedBHC(nodes):
            nonlocal merged_node_id

            def pick_random_subsample(nodes):
                n_nodes = len(nodes)
                selected_idx = np.random.choice(np.arange(n_nodes),
                                                size=int(np.ceil(
                                                    n_nodes*m_fraction)),
                                                replace=False)
                selected_nodes = [nodes[i] for i in selected_idx]

                return selected_nodes

            def filter_points(node, root):
                filtered_left = []
                filtered_right = []

                def compute_ppd(new_node, node):

                    data = self.data_model.compute_data(node.get_data())
                    new_data = self.data_model.compute_data(
                        new_node.get_data())

                    log_ppd = self.data_model.log_posterior_predictive(
                        new_data, data)

                    log_alpha_gamma_nk = math.log(
                        self.crp_alpha)+math.lgamma(node.get_count())

                    if not node.is_leaf():
                        log_prior = log_alpha_gamma_nk -\
                            logaddexp(log_alpha_gamma_nk,
                                      node.get_left().log_dk+node.get_right().log_dk)
                    else:
                        log_prior = 0

                    return log_prior+log_ppd

                for node in nodes:
                    left_ppd = compute_ppd(node, root.get_left())
                    right_ppd = compute_ppd(node, root.get_right())

                    if left_ppd > right_ppd:
                        filtered_left.append(node)
                    else:
                        filtered_right.append(node)

                return filtered_left, filtered_right

            left_subtree = []
            right_subtree = []
            while not left_subtree or not right_subtree:
                # pick fraction of data point by random
                selected_nodes = pick_random_subsample(nodes)

                if len(selected_nodes) < 3:
                    selected_nodes = nodes.copy()
                    bhc_ = bhc(selected_nodes, self.data_model, self.crp_alpha)
                    bhc_.fit()

                    if len(bhc_.nodes) > 1:

                        for i in range(len(nodes), len(bhc_.nodes)):
                            merged_node_id += 1
                            bhc_.nodes[i].id = merged_node_id

                            self.nodes.append(bhc_.nodes[i])
                            self.rks.append(math.exp(bhc_.nodes[i].log_rk))

                    return bhc_.root_node

                # train BHC model on fraction
                bhc_ = bhc(selected_nodes, self.data_model, self.crp_alpha)
                bhc_.fit()

                # Filter points
                left_subtree, right_subtree = filter_points(
                    nodes, bhc_.root_node)

            # run randomized algorithm
            left_bhc_root = randomizedBHC(left_subtree)

            right_bhc_root = randomizedBHC(right_subtree)

            merged_node_id += 1
            merged_node = self.create_merged_node(merged_node_id,
                                                  left_bhc_root,
                                                  right_bhc_root)

            self.nodes.append(merged_node)
            self.rks.append(math.exp(merged_node.log_rk))

            return merged_node

        self.root_node = randomizedBHC(self.nodes)
        self.omegas = self.compute_omegas(self.root_node)

    def get_Z(self):
        Z = []
        leaves_id_order = []

        n_leafs = int((len(self.nodes)+1)/2)
        ids_inner = iter(range(n_leafs, len(self.nodes)))
        ids_leafs = iter(range(n_leafs))
        inner_orig_id = []
        inner_new_id = []
        i = 1
        for node in self.nodes:
            if not node.is_leaf():
                inner_new_id.append(next(ids_inner))
                inner_orig_id.append(node.id)

                left = node.get_left()
                right = node.get_right()

                if left.is_leaf():
                    id_left = next(ids_leafs)
                    leaves_id_order.append(left.id)
                else:
                    id_left = inner_new_id[inner_orig_id.index(left.id)]

                if right.is_leaf():
                    id_right = next(ids_leafs)
                    leaves_id_order.append(right.id)
                else:
                    id_right = inner_new_id[inner_orig_id.index(right.id)]

                idx = [id_left,
                       id_right]

                Z.append([min(idx),
                          max(idx),
                          float(i),  # np.sqrt(node.get_count()),
                          node.get_count()])
                i += 1

        return Z, leaves_id_order

    @staticmethod
    def compute_omegas(node, log_ri=None, n_total=None):
        """ Recursive function to compute the mixture probabilites
            denoted omegas
        """

        log_ri = [] if log_ri is None else log_ri.copy()
        n_total = node.get_count() if n_total is None else n_total

        log_rk = node.log_rk

        log_omega = np.log(node.get_count())-np.log(n_total) + \
            log_rk+np.nan_to_num(np.sum(np.log(-np.expm1(np.array(log_ri)))))

        log_omega_node = {node.id: log_omega}

        log_ri.append(log_rk)

        if not node.is_leaf():
            log_omega_left = bhc.compute_omegas(
                node.get_left(), log_ri=log_ri, n_total=n_total)
            log_omega_right = bhc.compute_omegas(
                node.get_right(), log_ri=log_ri, n_total=n_total)
        else:
            log_omega_left = {}
            log_omega_right = {}

        log_omega_node.update(log_omega_left)
        log_omega_node.update(log_omega_right)
        return log_omega_node

    def predict(self, new_data, all_nodes=False):
        log_predictive_probs = []
        for node in self.nodes:
            nodes_data = node.get_data()
            data = self.data_model.compute_data(nodes_data)

            posterior_prob = self.data_model.log_posterior_predictive(
                new_data, data)
            log_predictive_probs.append(self.omegas[node.id]+posterior_prob)

        k_array = list(reversed(np.argsort(log_predictive_probs)))
        predict_prob = np.exp(log_predictive_probs)

        if not all_nodes:
            k = k_array[0]
            predict_prob = predict_prob[0]
        else:
            k = k_array

        return k, predict_prob

    def create_leaf_node(self, new_node_id, data, data_ids=None):

        data = self.data_model.compute_data(data)
        logp = self.data_model.log_marginal_likelihood(data)
        if logger.getEffectiveLevel() <= logging.DEBUG:
            values_to_check = (logp, math.log(self.crp_alpha))
        else:
            values_to_check = None
        log_dk = math.log(self.crp_alpha)
        log_ml = math.log(self.crp_alpha)+logp
        log_rk = 0
        data_ids_new = data_ids if data_ids is not None else [new_node_id]

        return Node(new_node_id, log_rk, log_ml, log_dk, data_ids=data_ids_new, data=data, values_to_check=values_to_check)

    def create_merged_node(self, new_node_id, left_node, right_node):

        def compute_log_alpha_gamma_nk(crp_alpha, nk):
            return math.log(crp_alpha) + math.lgamma(nk)

        def compute_posterior_merged(log_alpha_gamma_nk, log_marginal_h1, left_node, right_node):
            v1 = left_node.log_ml+right_node.log_ml
            v2 = log_alpha_gamma_nk+log_marginal_h1

            x = v1-v2

            return -(np.log(1+np.exp(-np.abs(x))) + np.maximum(x, 0))
            # return -np.log1p(np.exp(x))

        nk = left_node.get_count()+right_node.get_count()

        log_alpha_gamma_nk = compute_log_alpha_gamma_nk(self.crp_alpha, nk)

        nodes_data = left_node.get_data() + right_node.get_data()
        data = self.data_model.compute_data(nodes_data)

        log_marginal_h1 = self.data_model.log_marginal_likelihood(data)

        log_rk = compute_posterior_merged(
            log_alpha_gamma_nk, log_marginal_h1, left_node, right_node)

        if log_rk == -float('inf'):
            pdb.set_trace()

        # TODO: compute this recursively every time
        log_ml = logaddexp(log_alpha_gamma_nk+log_marginal_h1,
                           left_node.log_ml+right_node.log_ml)

        log_dk = logaddexp(log_alpha_gamma_nk,
                           left_node.log_dk+right_node.log_dk)

        if logger.getEffectiveLevel() <= logging.DEBUG:
            old_log_rk, old_log_ml, old_log_dk = self.compute_old_rk(
                left_node, right_node)

            assert (np.isclose(log_rk, old_log_rk)), \
                'log_rk(={}) is not equal to old_log_rk(={})'.format(
                log_rk, old_log_rk)

            values_to_check = (old_log_ml, old_log_dk)
        else:
            values_to_check = None

        return Node(new_node_id, log_rk, log_ml, log_dk, left_child=left_node, right_child=right_node, values_to_check=values_to_check)

    def compute_old_rk(self, left_node, right_node):

        def compute_log_alpha_gamma_nk(crp_alpha, nk):
            return math.log(crp_alpha) + math.lgamma(nk)

        def compute_log_dk(log_alpha_gamma_nk, log_children_dks):
            return logaddexp(log_alpha_gamma_nk, log_children_dks)

        def compute_log_pi(log_alpha_gamma_nk, log_dk):
            return -math.log(math.exp(log_dk - log_alpha_gamma_nk))

        def compute_posterior_merged(log_marginal, log_pi, left_node, right_node):
            log_numerator = log_pi + log_marginal
            log_1m_pi = math.log(-math.expm1(log_pi))
            log_denominator = logaddexp(
                log_numerator, log_1m_pi+left_node.values_to_check[0] + right_node.values_to_check[0])

            return log_numerator-log_denominator, log_denominator

        nk = left_node.get_count()+right_node.get_count()
        log_children_dks = left_node.values_to_check[1] + \
            right_node.values_to_check[1]

        log_alpha_gamma_nk = compute_log_alpha_gamma_nk(self.crp_alpha, nk)
        log_dk = compute_log_dk(log_alpha_gamma_nk, log_children_dks)
        log_pi = compute_log_pi(log_alpha_gamma_nk, log_dk)

        nodes_data = left_node.get_data() + right_node.get_data()
        data = self.data_model.compute_data(nodes_data)

        log_marginal = self.data_model.log_marginal_likelihood(data)

        log_rk, log_subtree = compute_posterior_merged(
            log_marginal, log_pi, left_node, right_node)

        return log_rk, log_subtree, log_dk

    def plot_dendrogram(self):
        colors = ['b' if np.exp(node.log_rk) >
                  0.5 else 'r' for node in self.nodes]
        Z, leaves_id_order = self.get_Z()
        if Z:
            Z = np.array(Z, ndmin=2)
            Z[:, 2] = 1.15**Z[:, 2]  # decent exp increase for visualization
            dend = dendrogram(Z, distance_sort=False, labels=leaves_id_order,
                              link_color_func=lambda k: colors[k])

    def plot_clusters(self, data=None):

        def plot_mean_cov(mean, covar, color, splot):
            v, w = linalg.eigh(covar)
            v = 2. * np.sqrt(2.) * np.sqrt(v)
            u = w[0] / linalg.norm(w[0])

            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1] / u[0])
            angle = 180. * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(
                mean[0], v[0], v[1], 180.+angle, color=color)
            # ell.set_clip_box(splot.bbox)
            ell.set_alpha(0.5)
            splot.add_artist(ell)

        if data is not None:
            leaves_id_order = [
                node.id for node in self.nodes if node.is_leaf()]
        else:
            leaves_id_order = [
                node.id for node in self.root_node.pre_order(lambda x: x)]

            data = np.vstack(self.root_node.get_data())

        top_cluster_nodes = self.get_cut_subtrees(
            self.root_node)

        colors = sns.color_palette(
            palette='deep', n_colors=len(top_cluster_nodes))
        colors = np.array(colors)

        colors_data = np.zeros((self.root_node.get_count(), 3))
        clusters = []
        mean_cov_tuples = []
        for k, node in enumerate(top_cluster_nodes):
            cluster_i_assignments = [leaves_id_order.index(i)
                                     for i in node.pre_order(lambda x: x.id)]
            clusters.append(cluster_i_assignments)

            colors_data[clusters[-1], :] = colors[k, :]

            if node.get_count() > 1:
                mean_cov_tuples.append((k, node.get_mean(),
                                        node.get_cov()))

        plt.scatter(data[:, 0], data[:, 1], c=colors_data)
        plt.scatter(data[-1, 0], data[-1, 1], color='w', s=12)

        for (i, mean, cov) in mean_cov_tuples:
            plot_mean_cov(mean, cov, colors[i, :], plt.gca())

    def compute_dendrogram_purity(self, true_clusters):
        # true_clusters is a list of list with ids of true clusters

        def flatten(l): return [item for sublist in l for item in sublist]

        def compute_purity(cluster_labels, true_labels):
            return len(set(cluster_labels+true_labels))/len(true_labels)

        true_clustering_pairs = [list(it.combinations(
            true_cluster, 2)) for true_cluster in true_clusters]

        n_true_pairs = sum(map(len, true_clustering_pairs))

        purity_sum = 0
        for k, pairs in tqdm(enumerate(true_clustering_pairs)):
            for pair in pairs:

                LCA = self.find_LCA(self.root_node, *pair)
                subtree_data_ids = flatten(LCA.pre_order(lambda x: x.data_ids))
                purity = compute_purity(subtree_data_ids, true_clusters[k])

                purity_sum += purity

        return (purity_sum/n_true_pairs, purity_sum, n_true_pairs)

    def find_LCA(self, root, p, q, search_func=lambda x: x.data_ids):

        def is_potential_LCA(node, p, q):
            leaves = [item for l in node.pre_order(search_func) for item in l]
            return p in leaves and q in leaves

        left_child = root.get_left()
        right_child = root.get_right()

        if not left_child.is_leaf() and is_potential_LCA(left_child, p, q):
            LCA = self.find_LCA(root.get_left(), p, q)
        elif not right_child.is_leaf() and is_potential_LCA(right_child, p, q):
            LCA = self.find_LCA(root.get_right(), p, q)
        else:
            LCA = root

        return LCA

    @staticmethod
    def get_cut_subtrees(node, nodelist=None):
        nodelist = [] if nodelist is None else nodelist

        if np.exp(node.log_rk) < 0.5:
            left = bhc.get_cut_subtrees(node.get_left())
            right = bhc.get_cut_subtrees(node.get_right())
            return nodelist+left+right
        else:
            return [node]


class Node(ClusterNode):
    """
    Based off scipy's ClusterNode class
    """

    def __init__(self, node_id, log_rk, log_ml, log_dk, data_ids=None, data=None, count=None, left_child=None, right_child=None, values_to_check=None):

        if left_child is not None:
            count = left_child.count+right_child.count
        else:
            count = count if count else 1

        self.log_rk = log_rk
        self.log_ml = log_ml
        self.log_dk = log_dk
        self.data = data
        self.data_ids = data_ids
        self.values_to_check = values_to_check

        super().__init__(node_id, left=left_child, right=right_child, count=count)

    def __str__(self, level=0, sym=""):
        new_sym = "|" if np.exp(self.log_rk) > 0.5 else "x"
        if self.get_right():
            ret = self.get_right().__str__(level+1, new_sym)
        else:
            ret = ""

        ret += "   "*(level-1)+(sym+"--")*(level > 0)+str(self.id)+"\n"

        if self.get_left():
            ret += self.get_left().__str__(level+1, new_sym)

        return ret

    def get_data(self):
        return self.pre_order(lambda x: x.data)

    def get_mean(self):
        data = self.get_data()
        return np.mean(np.vstack(data), axis=0, keepdims=True) \
            if type(data[0]) is np.ndarray \
            else sum(data).get_mean()

    def get_cov(self):
        data = self.get_data()
        return np.cov(np.vstack(data), rowvar=False) \
            if type(data[0]) is np.ndarray \
            else sum(data).get_cov()
