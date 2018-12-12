from __future__ import print_function, division
import itertools as it
import numpy as np
import sys
from scipy.cluster.hierarchy import ClusterNode

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
        self.Z = []

        current_roots = assignment

        while len(current_roots) > 1:
            max_rk = float('-Inf')
            merged_node = None
            merged_node_id = len(self.nodes)

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

            self.Z.append([self.nodes[merged_left].id, self.nodes[merged_right].id,
                           float(merged_node.id), merged_node.get_count()])

        self.assignments = np.array(self.assignments)
        self.root_node = self.nodes[-1]

        self.omegas = self.compute_omegas(self.root_node)

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

        nodes_data = left_node.pre_order(lambda x: x.data) + \
            right_node.pre_order(lambda x: x.data)

        data = self.data_model.compute_data(nodes_data)

        logp = self.data_model.log_marginal_likelihood(data)
        numer = log_pi + logp

        neg_pi = math.log(-math.expm1(log_pi))
        log_ml = logaddexp(numer, neg_pi+left_node.log_ml + right_node.log_ml)

        log_rk = numer-log_ml

        if log_pi == 0:
            raise RuntimeError('Precision error')

        return Node(new_node_id, log_rk, log_ml, log_dk, left_child=left_node, right_child=right_node)


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
