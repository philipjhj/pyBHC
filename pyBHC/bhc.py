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

        if not all(isinstance(n, Node) for n in data):
            self.nodes = [self.create_leaf_node(
                node_id, np.array([x])) for node_id, x in enumerate(data)]
        else:
            self.nodes = data

    def fit(self):
        # initialize the tree

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

            self.nodes.append(merged_node)

            self.rks.append(math.exp(max_rk))

            for i, k in enumerate(assignment):
                if k == merged_right:
                    assignment[i] = merged_left
            self.assignments.append(list(assignment))

            self.Z.append([self.nodes[merged_left].id, self.nodes[merged_right].id,
                           float(merged_node.id), merged_node.get_count()])

        self.root_node = self.nodes[0]
        self.assignments = np.array(self.assignments)

        # The denominator of log_rk is at the final merge is an
        # estimate of the marginal likelihood of the data under DPMM
        self.lml = self.root_node.log_ml

    def find_path(self, index):
        """ find_path(index)

            Finds the sequence of left and right merges needed to
            run from the root node to a particular leaf.

            Parameters
            ----------
            index : int
                The index of the leaf for which we want the path
                from the root node.
        """
        merge_path = []
        last_leftmost_index = self.assignments[-1][index]
        last_right_incluster = (self.assignments[-1]
                                == last_leftmost_index)

        for it in range(len(self.assignments)-2, -1, -1):
            new_leftmost_index = self.assignments[it][index]

            if new_leftmost_index != last_leftmost_index:
                # True if leaf is on the right hand side of a merge
                merge_path.append("right")
                last_leftmost_index = new_leftmost_index
                last_right_incluster = (self.assignments[it]
                                        == new_leftmost_index)

            else:       # Not in a right hand side of a merge

                new_right_incluster = (self.assignments[it]
                                       == last_leftmost_index)

                if (new_right_incluster != last_right_incluster).any():
                    # True if leaf is on the left hand side of a merge
                    merge_path.append("left")
                    last_right_incluster = new_right_incluster

        return merge_path

    def sample(self, size=1):

        output = np.zeros((size, self.root_node.data.shape[1]))

        for it in range(size):

            sampled = False
            node = self.root_node

            while not sampled:

                if node.log_rk is None:     # Node is a leaf
                    output[it, :] = self.data_model.conditional_sample(
                        node.data)
                    sampled = True

                elif np.random.rand() < math.exp(node.log_rk):
                    # sample from node
                    output[it, :] = self.data_model.conditional_sample(
                        node.data)
                    sampled = True

                else:   # drop to next level
                    child_ratio = (node.left_child.nk
                                   / (node.left_child.nk+node.right_child.nk))
                    if np.random.rand() >= child_ratio:
                        node = node.right_child
                    else:
                        node = node.left_child

        return output

    def create_leaf_node(self, new_node_id, data):

        data = self.data_model.compute_data(data)
        log_dk = math.log(self.crp_alpha)
        logp = self.data_model.log_marginal_likelihood(data)
        log_ml = logp
        log_rk = None

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
