from __future__ import print_function, division
import itertools as it
import numpy as np
import sys
from scipy.cluster.hierarchy import ClusterNode, dendrogram
from scipy import linalg
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import copy
from collections import deque

import matplotlib as mpl
from numpy import logaddexp
import math
import time

import pdb
from pudb import set_trace


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

    def __init__(self, data, data_model, crp_alpha=1.0, cache_iteration_run_time=True):
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
        self.cache_iteration_run_time = cache_iteration_run_time
        self.iteration_run_times = []
        self.data_model = data_model
        self.crp_alpha = crp_alpha

        # initialize the tree
        if not all(isinstance(n, Node) for n in data):
            self.nodes = [self.create_leaf_node(
                node_id, np.array([x])) for node_id, x in enumerate(data)]
        else:
            self.nodes = data

    def fit(self, verbose=True):
        n_data = len(self.nodes)
        assignment = list(range(n_data))
        self.assignments = [list(assignment)]
        self.rks = []

        current_roots = assignment
        cached_nodes = dict()
        merged_node_id = max([node.id for node in self.nodes])

        i_nodes = 1
        while len(current_roots) > 1:
            if self.cache_iteration_run_time:
                t_a = time.time()
            if verbose:
                logger.info(
                    'Merging next node [{}/{}]'.format(i_nodes, n_data))
                i_nodes += 1
            max_rk = float('-Inf')
            merged_node = None

            # for each pair of clusters (nodes), compute the merger
            # score.

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

            if self.cache_iteration_run_time:
                self.iteration_run_times.append(time.time()-t_a)

        self.assignments = np.array(self.assignments)
        self.root_node = self.nodes[-1]

        self.omegas = self.compute_omegas()

    def randomized_fit(self, m=0.2):

        merged_node_id = max([node.id for node in self.nodes])
        self.rks = []
        n_nodes = len(self.nodes)
        n_depth = np.ceil(np.log2(n_nodes))

        if isinstance(m, int):
            n_select = m
        elif isinstance(m, float):
            n_select = max(2, int(np.ceil(len(n_nodes)*m)))
        else:
            logger.info('m should be an int or a float')

        def randomizedBHC(nodes, i_depth=0):
            n_nodes_local = len(nodes)
            nonlocal merged_node_id

            def select_n_nodes(n, nodes):
                selected_idx = np.random.choice(np.arange(len(nodes)),
                                                size=n,
                                                replace=False)

                nodes_selected = [nodes[i] for i in selected_idx]
                nodes_remaining = [nodes[i] for i in range(len(nodes))
                                   if i not in selected_idx]

                return nodes_selected, nodes_remaining

            def filter_points(nodes_to_be_filtered, root, omegas):
                filtered_left = root.get_left().get_leaves()
                filtered_right = root.get_right().get_leaves()

                def compute_subtree_probability(new_node, subtree_root, omega):

                    data = self.data_model.compute_data(
                        subtree_root.get_data())
                    new_data = self.data_model.compute_data(
                        new_node.get_data())

                    # Compute subtree prior

                    if not subtree_root.is_leaf():
                        log_alpha_gamma_nk = math.log(
                            self.crp_alpha)+math.lgamma(subtree_root.get_count())

                        log_prior = log_alpha_gamma_nk -\
                            logaddexp(log_alpha_gamma_nk,
                                      subtree_root.get_left().log_dk
                                      + subtree_root.get_right().log_dk)
                    else:
                        log_prior = 0

                    new_node = self.create_leaf_node(1, new_data)
                    tmp_node = self.create_merged_node(0,
                                                       subtree_root,
                                                       new_node)

                    log_prob = log_prior+tmp_node.log_rk

                    return log_prob

                for node in nodes_to_be_filtered:
                    l_subtree_prob = compute_subtree_probability(
                        node, root.get_left(), omegas[root.get_left().id])
                    r_subtree_prob = compute_subtree_probability(
                        node, root.get_right(), omegas[root.get_right().id])

                    if l_subtree_prob > r_subtree_prob:
                        filtered_left.append(node)
                    else:
                        filtered_right.append(node)

                return filtered_left, filtered_right

            logger.info('Fitting {:.0f} nodes at depth [{:.0f}] with nodes [{}/{}]'.format(
                n_nodes_local, i_depth, n_select, n_nodes))

            # if n_select < n_nodes_local:
            # pick fraction of data point by random
            nodes_selected, nodes_remaining = select_n_nodes(
                n_select, nodes)

            # train BHC model on fraction
            assert len(nodes_selected) == n_select
            bhc_filter = bhc(copy.deepcopy(nodes_selected),
                             self.data_model, self.crp_alpha)
            bhc_filter.fit(verbose=False)
            assert len(nodes_selected) == n_select

            # Filter points
            l_subtree, r_subtree = filter_points(
                copy.deepcopy(nodes_remaining), bhc_filter.root_node, bhc_filter.omegas)

            subtree_roots = []
            for subtree in [l_subtree, r_subtree]:
                if len(subtree) > n_select:
                    subtree_root = randomizedBHC(
                        copy.deepcopy(subtree), i_depth=i_depth+1)

                elif len(subtree) == 1:
                    subtree_root = subtree[0]
                elif 1 < len(subtree) and len(subtree) <= n_select:
                    logger.info(
                        'Subtree is smaller than m; fitting final bhc.')
                    bhc_ = bhc(copy.deepcopy(subtree),
                               self.data_model, self.crp_alpha)
                    bhc_.fit(verbose=False)

                    for i in range(len(subtree), len(bhc_.nodes)):
                        merged_node_id += 1
                        bhc_.nodes[i].id = merged_node_id

                        self.nodes.append(bhc_.nodes[i])
                        self.rks.append(math.exp(bhc_.nodes[i].log_rk))
                        logger.debug(
                            'Adds node [{} / {}]'.format(len(self.nodes), 2*n_nodes-1))

                    subtree_root = bhc_.root_node
                else:
                    logger.debug(
                        "Bad (empty?) subtree in random algorithm; this should not happen.")

                subtree_roots.append(subtree_root)

            merged_node_id += 1
            merged_node = self.create_merged_node(merged_node_id,
                                                  subtree_roots[0],
                                                  subtree_roots[1])

            self.nodes.append(merged_node)
            self.rks.append(math.exp(merged_node.log_rk))
            logger.debug(
                'Adds node [{} / {}]'.format(len(self.nodes), 2*n_nodes-1))

            return merged_node

        logger.info(
            'Total nodes [{} / {}]'.format(len(self.nodes), 2*n_nodes-1))
        self.root_node = randomizedBHC(self.nodes)
        self.omegas = self.compute_omegas()

    def get_Z(self):

        Z = []
        leaves_id_order = []

        n_nodes = len(self.nodes)
        n_leafs = int((n_nodes+1)/2)
        ids_inner = iter(range(n_leafs, len(self.nodes)))
        ids_leafs = iter(range(n_leafs))
        inner_orig_id = []
        inner_new_id = []

        colors = list('b')*n_nodes

        i = 1
        for node in self.nodes:
            if not node.is_leaf():
                inner_new_id.append(next(ids_inner))
                inner_orig_id.append(node.id)

                if np.exp(node.log_rk) < 0.5:
                    colors[inner_new_id[-1]] = 'r'

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

        return Z, leaves_id_order, colors

    def get_flat_clusters(self):
        """ Get flat cluster assignments

        """

        data = np.vstack(self.root_node.get_data())

        top_cluster_nodes = self.get_cut_subtrees(self.root_node)

        flat_clusters = []
        for k, node in enumerate(top_cluster_nodes):
            cluster_i_assignments = node.pre_order(lambda x: x.id)

            if node.get_count() > 1:
                mean_cov = (node.get_mean(), node.get_cov())
            else:
                mean_cov = (None, None)

            flat_clusters.append(
                {'mean_cov': mean_cov, 'data_ids': cluster_i_assignments})

        return data, flat_clusters

    def compute_omegas(self):
        """ Non-Recursive function to compute the mixture probabilites
            denoted omegas
        """

        def get_log_omega(log_rk, log_ri):
            return log_rk + np.nan_to_num(np.sum(np.log(-np.expm1(np.array(log_ri)))))

        q_nodes = deque()

        log_ri = []
        q_nodes.append((self.root_node, log_ri))

        log_omegas = {}
        while q_nodes:
            next_node_items = q_nodes.popleft()

            node = next_node_items[0]
            log_ri = next_node_items[1]

            log_rk = node.log_rk

            log_omega_node = {node.id: get_log_omega(log_rk, log_ri)}
            log_omegas.update(log_omega_node)

            left_child = node.get_left()
            right_child = node.get_right()

            if left_child:
                q_nodes.append((left_child, log_ri+[log_rk]))
            if right_child:
                q_nodes.append((right_child, log_ri+[log_rk]))

        return log_omegas

    def predict(self, new_data, all_nodes=False):
        """ Computes the probability of new data belonging
        to each node using the posterior predictive distribution

        if `all_nodes` is true, then returns two lists sorted by
        most likely node to least likely node one with the ordering
        in the nodes attribute, and one with the corresponding
        probabilities, otherwise only the most likely node
        """

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
            set_trace()

        # TODO: compute this recursively every time
        log_ml = logaddexp(log_alpha_gamma_nk+log_marginal_h1,
                           left_node.log_ml+right_node.log_ml)

        log_dk = logaddexp(log_alpha_gamma_nk,
                           left_node.log_dk+right_node.log_dk)

        if logger.getEffectiveLevel() <= logging.DEBUG:
            old_log_rk, old_log_ml, old_log_dk, valid_computation = self.compute_old_rk(
                left_node, right_node)

            if valid_computation:
                assert (np.isclose(log_rk, old_log_rk)), \
                    'log_rk(={}) is not equal to old_log_rk(={})'.format(
                    log_rk, old_log_rk)
            else:
                logger.warning(
                    'Numerically issues when computing old log_rk values; skipping test')

            values_to_check = (old_log_ml, old_log_dk)
        else:
            values_to_check = None

        return Node(new_node_id, log_rk, log_ml, log_dk, left_child=left_node, right_child=right_node, values_to_check=values_to_check)

    def compute_old_rk(self, left_node, right_node):

        valid_computation = True

        def compute_log_alpha_gamma_nk(crp_alpha, nk):
            return math.log(crp_alpha) + math.lgamma(nk)

        def compute_log_dk(log_alpha_gamma_nk, log_children_dks):
            return logaddexp(log_alpha_gamma_nk, log_children_dks)

        def compute_log_pi(log_alpha_gamma_nk, log_dk):
            return -math.log(math.exp(log_dk - log_alpha_gamma_nk))

        def compute_posterior_merged(log_marginal, log_pi, left_node, right_node):
            nonlocal valid_computation

            log_numerator = log_pi + log_marginal
            try:
                log_1m_pi = math.log(-math.expm1(log_pi))
            except ValueError:
                log_1m_pi = 0
                valid_computation = False

            log_denominator = logaddexp(
                log_numerator, log_1m_pi+left_node.values_to_check[0] + right_node.values_to_check[0])

            return log_numerator-log_denominator, log_denominator

        # set_trace()
        nk = left_node.get_count()+right_node.get_count()
        log_children_dks = left_node.values_to_check[1] + \
            right_node.values_to_check[1]

        log_alpha_gamma_nk = compute_log_alpha_gamma_nk(self.crp_alpha, nk)
        log_dk = compute_log_dk(log_alpha_gamma_nk, log_children_dks)
        log_pi = compute_log_pi(log_alpha_gamma_nk, log_dk)

        if np.isclose(log_dk, log_alpha_gamma_nk):
            valid_computation = False

        nodes_data = left_node.get_data() + right_node.get_data()
        data = self.data_model.compute_data(nodes_data)

        log_marginal = self.data_model.log_marginal_likelihood(data)

        log_rk, log_subtree = compute_posterior_merged(
            log_marginal, log_pi, left_node, right_node)

        return log_rk, log_subtree, log_dk, valid_computation

    def plot_dendrogram(self):
        Z, leaves_id_order, colors = self.get_Z()

        if Z:
            Z = np.array(Z, ndmin=2)
            Z[:, 2] = 1.15**Z[:, 2]  # decent exp increase for visualization
            dend = dendrogram(Z, distance_sort=False, labels=leaves_id_order,
                              link_color_func=lambda k: colors[k])

    def plot_clusters(self):

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

        data, flat_clusters = self.get_flat_clusters()

        idx_new_node = np.argmax(self.root_node.pre_order(lambda x: x.id))

        colors = sns.color_palette(
            palette='deep', n_colors=len(flat_clusters))
        colors = np.array(colors)

        colors_data = colors[[k for k in range(len(flat_clusters))
                              for _ in range(len(flat_clusters[k]['data_ids']))], :]

        if data is not None:
            plt.scatter(data[:, 0], data[:, 1], c=colors_data)
            plt.scatter(data[idx_new_node, 0],
                        data[idx_new_node, 1], color='w', s=12)

        for i, cluster in enumerate(flat_clusters):
            mean = cluster['mean_cov'][0]
            cov = cluster['mean_cov'][1]

            if mean is not None and cov is not None:
                plot_mean_cov(mean, cov, colors[i, :], plt.gca())

    def plot_bar_tree(self):

        def get_values(nodes, values=None):

            values = values if values else []

            values_nodes = []
            children_nodes = []

            for node in nodes:
                values_nodes.append(np.exp(node.log_rk))
                if node.get_left():
                    children_nodes.append(node.get_left())
                if node.get_right():
                    children_nodes.append(node.get_right())

            values.append(values_nodes)

            if children_nodes:
                values = get_values(children_nodes, values)

            return values

        values_all_levels = get_values([self.root_node])

        height = len(values_all_levels)
        k = 0
        for i_level, values_level in enumerate(values_all_levels):
            width = len(values_level)
            for i, values_node in enumerate(values_level):
                plt.subplot(height, width, i_level*width+i+1)
                plt.bar(1, values_node)
                plt.ylim([0, 1])

        # plt.show()
        # set_trace()

    def compute_dendrogram_purity(self, true_clusters):
        # true_clusters is a list of list with ids of true clusters

        def flatten(l): return [item for sublist in l for item in sublist]

        def compute_purity(cluster_labels, true_labels):
            return len(set(cluster_labels+true_labels))/len(true_labels)

        true_clustering_pairs = [list(it.combinations(
            true_cluster, 2)) for true_cluster in true_clusters]

        n_true_pairs = sum(map(len, true_clustering_pairs))

        purity_sum = 0
        for k, pairs in enumerate(true_clustering_pairs):
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
            if node.get_left():
                left = bhc.get_cut_subtrees(node.get_left())
            else:
                left = []
            if node.get_right():
                right = bhc.get_cut_subtrees(node.get_right())
            else:
                right = []
            return nodelist+left+right
        else:
            return [node]

    def get_all_nodes_from_tree(self):
        cluster_roots = self.get_cut_subtrees(self.root_node)

        all_nodes = []

        def get_nodes(node, all_nodes=None):

            all_nodes = all_nodes if all_nodes else []

            all_nodes.append(node)
            if node.get_left():
                all_nodes = get_nodes(node.get_left(), all_nodes)
            if node.get_left():
                all_nodes = get_nodes(node.get_right(), all_nodes)

            return all_nodes

        return list(map(get_nodes, cluster_roots))


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

    def get_leaves(self):
        return self.pre_order(lambda x: x)

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
#
#    def get_gradient(self, crp_alpha):
#        if self.is_leaf():
#            # TODO: for collapsed nodes, this shouldn't return 0
#            return 0
#        else:
#            left_child = self.get_left()
#            dk_left = np.exp(left_child.log_dk)
#            right_child = self.get_right()
#            dk_right = np.exp(right_child.log_dk)
#
#            # computing this might require some tricks
#            pi = crp_alpha*gamma(self.get_count())/(crp_alpha*gamma(self.get_count())
#                                                    + dk_left*dk_right)
#            grad_dk = gamma(self.get_count())+left_child.grad_dk * \
#                dk_right+dk_left*right_child.grad_dk
#            grad_pi = pi/crp_alpha-pi/np.exp(self.log_dk)*grad_dk
#
#            # TODO: get the missing values here
#            log_marginal_h1 =
#            grad_left_child = left_child.get_gradient(crp_alpha)
#            log_marginal_tree_right_child =
#            grad_right_child = right_child.get_gradient(crp_alpha)
#            log_marginal_tree_left_child =
