from __future__ import print_function, division
import itertools as it
import numpy as np
import sys
from scipy.cluster.hierarchy import ClusterNode, dendrogram
from scipy import linalg
import matplotlib.pyplot as plt

import matplotlib as mpl
from numpy import logaddexp
import math


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

        while len(current_roots) > 1:
            max_rk = float('-Inf')
            merged_node = None
            merged_node_id = max([node.id for node in self.nodes])+1

            # for each pair of clusters (nodes), compute the merger
            # score.
            for left_idx, right_idx in it.combinations(current_roots, 2):
                tmp_node = self.create_merged_node(merged_node_id,
                                                   self.nodes[left_idx],
                                                   self.nodes[right_idx])

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

            self.rks.append(math.exp(max_rk))

            for i, k in enumerate(assignment):
                if k == merged_right:
                    assignment[i] = merged_left
            self.assignments.append(list(assignment))

        self.assignments = np.array(self.assignments)
        self.root_node = self.nodes[-1]

        self.omegas = self.compute_omegas(self.root_node)

    def get_Z(self):
        Z = []
        #ids = [node.id for node in self.nodes]
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
                else:
                    id_left = inner_new_id[inner_orig_id.index(left.id)]

                if right.is_leaf():
                    id_right = next(ids_leafs)
                else:
                    id_right = inner_new_id[inner_orig_id.index(right.id)]

                idx = [id_left,
                       id_right]

                Z.append([min(idx),
                          max(idx),
                          float(i),  # np.sqrt(node.get_count()),
                          node.get_count()])
                i += 1

        return Z

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

        log_omega_node = [log_omega]

        log_ri.append(log_rk)

        if not node.is_leaf():
            log_omega_left = bhc.compute_omegas(
                node.get_left(), log_ri=log_ri, n_total=n_total)
            log_omega_right = bhc.compute_omegas(
                node.get_right(), log_ri=log_ri, n_total=n_total)
        else:
            log_omega_left = []
            log_omega_right = []

        return log_omega_node+log_omega_left+log_omega_right

    def predict(self, new_data, all_nodes=False):
        log_predictive_probs = []
        for i, node in enumerate(self.nodes):
            nodes_data = node.get_data()
            data = self.data_model.compute_data(nodes_data)

            posterior_prob = self.data_model.log_posterior_predictive(
                new_data, data)
            log_predictive_probs.append(self.omegas[i]+posterior_prob)

        k_array = np.argsort(log_predictive_probs)
        predict_prob = np.sum(np.exp(log_predictive_probs))

        if not all_nodes:
            k = k_array[0]
        else:
            k = k_array

        return k, predict_prob

    def create_leaf_node(self, new_node_id, data):

        data = self.data_model.compute_data(data)
        log_dk = math.log(self.crp_alpha)
        logp = self.data_model.log_marginal_likelihood(data)
        log_ml = logp
        log_rk = 0

        return Node(new_node_id, log_rk, log_ml, log_dk, data=data)

    def create_merged_node(self, new_node_id, left_node, right_node):

        nk = left_node.get_count()+right_node.get_count()
        dk_sum = left_node.log_dk + right_node.log_dk

        log_dk = logaddexp(math.log(self.crp_alpha) + math.lgamma(nk),
                           dk_sum)
        log_pi = -math.log1p(math.exp(dk_sum - math.log(self.crp_alpha)
                                      - math.lgamma(nk)))

        # Calculate log_rk - the log probability of the merge

        nodes_data = left_node.get_data() + right_node.get_data()

        data = self.data_model.compute_data(nodes_data)

        logp = self.data_model.log_marginal_likelihood(data)
        numer = log_pi + logp

        neg_pi = math.log(-math.expm1(log_pi))
        log_ml = logaddexp(numer, neg_pi+left_node.log_ml + right_node.log_ml)

        log_rk = numer-log_ml

        if log_pi == 0:
            raise RuntimeError('Precision error')

        return Node(new_node_id, log_rk, log_ml, log_dk, left_child=left_node, right_child=right_node)

    def plot_dendrogram(self):
        colors = ['b' if np.exp(node.log_rk) >
                  0.5 else 'r' for node in self.nodes]
        Z = self.get_Z()
        if Z:
            Z = np.array(Z, ndmin=2)
            Z[:, 2] = 1.15**Z[:, 2]  # decent exp increase for visualization
            leaf_ids = self.root_node.pre_order(lambda x: x.id)
            dend = dendrogram(Z, distance_sort=True, labels=leaf_ids,
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

        top_cluster_nodes = self.get_cut_subtrees(
            self.root_node)

        leaf_ids = [node.id for node in self.nodes if node.is_leaf()]
        clusters = [[leaf_ids.index(i) for i in node.pre_order(lambda x: x.id)]
                    for node in top_cluster_nodes]

        clusters_id = [node.pre_order(lambda x: x.id)
                       for node in top_cluster_nodes]

        data_colors = np.zeros(self.root_node.get_count())

        for i, cluster in enumerate(clusters):
            data_colors[cluster] = int(i)

        data_colors = data_colors[np.argsort(leaf_ids)]

        # TODO: get n colors
        # colors = plt.cm.get_cmap('hsv', max(len(clusters)+1, 4))
        # plot_colors = [colors(i) for i in data_colors]

        colors = 'rbycmg'
        plot_colors = [colors[int(i)] for i in data_colors]

        data = np.vstack(self.root_node.get_data()
                         ) if data is None else data

        plt.scatter(data[:, 0], data[:, 1], color=plot_colors)
        plt.title(clusters_id)

        plt.scatter(data[-1, 0], data[-1, 1], color='w', s=12)

        for i, cluster_node in enumerate(top_cluster_nodes):
            data_n = cluster_node.get_count()
            mean = cluster_node.get_mean()
            cov = cluster_node.get_cov()
            if data_n > 2:
                plot_mean_cov(mean, cov, colors[i], plt.gca())

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

    def __init__(self, node_id, log_rk, log_ml, log_dk, data=None, left_child=None, right_child=None):

        if left_child is not None:
            count = left_child.count+right_child.count
        else:
            count = 1

        self.log_rk = log_rk
        self.log_ml = log_ml
        self.log_dk = log_dk
        self.data = data

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
