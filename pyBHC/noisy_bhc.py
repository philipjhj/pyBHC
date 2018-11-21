from __future__ import print_function, division
import itertools as it
import numpy as np
import sys

from numpy import logaddexp
import math

from . import gmm


class noisy_bhc(object):
    """
    An instance of Bayesian hierarchical clustering CRP mixture model
    for data observed with (Gaussian) noise.
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
    Nleaves : int
        The number of initial  'leaf' nodes formed. If constructing
        directly from data this is the same as the number of data
        points. If building from preclustered data this is the
        number of pre-clusters
    Ndata : int
        The total number of data points
    Notes
    -----
    The cost of BHC scales as O(n^2) and so becomes inpractically
    large for datasets of more than a few hundred points.
    """

    def __init__(self, data_model, crp_alpha=1.0, verbose=False):
        self.data_model = data_model
        self.crp_alpha = crp_alpha

        self.verbose = verbose

        # initialise the posterior & cavity GMMs
        self.post_GMMs = None
        self.global_GMM = None
        self.cavity_GMMs = None

    def init_tree(self, data, data_uncerts):
        """ init_tree()

            Create the nodes from which a BHC tree can be built.
            Each datum is initially assigned to its own node.

            Parameters
            ----------
            data : numpy.ndarray (n, d)
                Array of data where each row is a data point and each
                column is a dimension.
            data_uncerts: numpy.ndarray (n, d, d)
                Array of uncertainties on the data, such that the first
                axis is a data point and the second two axes are for the
                covariance matrix.
            Returns
            -------
            nodes : list(noisy_Node)
                A list of nodes each of which contains a single datum
        """
        self.data = data
        self.data_uncerts = data_uncerts
        self.Nleaves = self.data.shape[0]
        self.Ndata = self.data.shape[0]

        # create nodes
        nodes = dict((i, noisy_Node(np.array([x]),
                                    np.array([data_uncerts[i]]),
                                    self.data_model, self.crp_alpha,
                                    indexes=i))
                     for i, x in enumerate(data))

        return nodes

    def init_preclustered(self, data, data_uncerts):
        """ init_preclustered(data, data_uncerts)

            Create the nodes from which a BHC tree can be built.
            The data have been pre-clustered so that each initial
            node contains some non-zero number of data

            Parameters
            ----------
            data : list(numpy.ndarray (n, d))
                Array of data where each row is a data point and each
                column is a dimension.
            data_uncerts: list(numpy.ndarray (n, d, d))
                Array of uncertainties on the data, such that the first
                axis is a data point and the second two axes are for the
                covariance matrix.
            Returns
            -------
            nodes : list(noisy_Node)
                A list of nodes each of which contains a single datum
        """
        # Assemble data into one big array
        self.data = np.concatenate(data)
        self.data_uncerts = np.concatenate(data_uncerts)
        self.Nleaves = len(data)
        self.Ndata = self.data.shape[0]

        # create nodes

        nodes = {}
        start_ind = 0
        for i in range(len(data)):
            nodes[i] = noisy_Node(data[i], data_uncerts[i], self.data_model,
                                  self.crp_alpha,
                                  indexes=range(start_ind,
                                                start_ind + data[i].shape[0]))
            start_ind += data[i].shape[0]

        return nodes

    def cluster_nodes(self, nodes):
        """ cluster_nodes()

            Perform aglomorative BHC clustering on the nodes in
            self.nodes

            Parameters
            ----------
            nodes : list(noisy_Node)
                A list of nodes to be merged

            Returns
            -------
            None
        """
        n_nodes = len(nodes)
        start_n_nodes = len(nodes)

        self.assignments = [[i for i in range(self.Nleaves)]]

        assignment = self.assignments[0][:]

        while n_nodes > 1:
            if self.verbose:
                sys.stdout.write("\r{0:d} of {1:d} ".format(n_nodes,
                                                            start_n_nodes))
                sys.stdout.flush()

            max_rk = float('-Inf')
            merged_node = None

            # for each pair of clusters (nodes), compute the merger
            # score.
            for left_idx, right_idx in it.combinations(nodes.keys(), 2):
                tmp_node = noisy_Node.as_merge(nodes[left_idx],
                                               nodes[right_idx])

                if tmp_node.log_rk >= max_rk:
                    max_rk = tmp_node.log_rk
                    merged_node = tmp_node
                    merged_right = right_idx
                    merged_left = left_idx

            # Merge the highest-scoring pair
            del nodes[merged_right]
            nodes[merged_left] = merged_node

            for i, k in enumerate(assignment):
                if k == merged_right:
                    assignment[i] = merged_left
            self.assignments.append(list(assignment))

            n_nodes -= 1

        self.root_node = nodes[0]
        self.assignments = np.array(self.assignments)

        # travese tree setting params
        self.set_params(self.root_node)

        # The denominator of log_rk is at the final merge is an
        # estimate of the marginal likelihood of the data under DPMM
        self.lml = self.root_node.log_ml

    @classmethod
    def from_data(cls, data, data_uncerts, data_model, crp_alpha=1.0,
                  verbose=False):
        """
        Init a bhc instance and perform the clustering given some
        unclustered data.

        Parameters
        ----------
        data : numpy.ndarray (n, d)
            Array of data where each row is a data point and each
            column is a dimension.
        data_uncerts: numpy.ndarray (n, d, d)
            Array of uncertainties on the data, such that the first
            axis is a data point and the second two axes are for the
            covariance matrix.
        data_model : CollapsibleDistribution
            Provides the approprite ``log_marginal_likelihood``
            function for the data.
        crp_alpha : float (0, Inf)
            CRP concentration parameter.
        verbose : bool, optional
            Determibes whetrher info gets dumped to stdout.
        """
        bhc = cls(data_model, crp_alpha, verbose)
        nodes = bhc.init_tree(data, data_uncerts)
        bhc.cluster_nodes(nodes)

        return bhc

    @classmethod
    def from_preclustered(cls, data, data_uncerts, data_model, crp_alpha=1.0,
                          verbose=False):
        """
        Init a bhc instance and perform the clustering given some
        unclustered data.

        Parameters
        ----------
        data : list(numpy.ndarray (n, d))
            Array of data where each row is a data point and each
            column is a dimension.
        data_uncerts: list(numpy.ndarray (n, d, d))
            Array of uncertainties on the data, such that the first
            axis is a data point and the second two axes are for the
            covariance matrix.
        data_model : CollapsibleDistribution
            Provides the approprite ``log_marginal_likelihood``
            function for the data.
        crp_alpha : float (0, Inf)
            CRP concentration parameter.
        verbose : bool, optional
            Determibes whetrher info gets dumped to stdout.
        """
        bhc = cls(data_model, crp_alpha, verbose)
        nodes = bhc.init_preclustered(data, data_uncerts)
        bhc.cluster_nodes(nodes)

        return bhc

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

    def get_global_posterior(self):
        """ get_global_posteriors()

            Find the posterior implied by the clustering as a Gaussian
            mixture, with each component in he mixture corresponding
            to a node in the clustering.

        """
        # initialise a GMM
        self.global_GMM = gmm.GMM()
        self.global_posterior_preds = []

        # Traverse tree

        self.add_node_posterior(self.root_node, self.global_GMM,
                                self.global_posterior_preds, recurse=True)

        self.global_GMM.normalise_weights()
        self.global_GMM.set_mean_covar()

    def add_node_posterior(self, node, GMM, posterior_preds=None,
                           weight_mult=1., recurse=False):

        if node.log_rk is not None:
            weight = (node.prev_wk * math.exp(node.log_rk)
                      * node.data.shape[0]/self.Ndata)
        else:           # a leaf
            weight = node.prev_wk * node.data.shape[0]/self.Ndata
        weight *= weight_mult
        mu = node.params[0]
        sigma = node.params[1] + node.params[2]

        GMM.add_component(weight, mu, sigma)

        if posterior_preds is not None:
            posterior_preds.append(
                node.data_model.freeze_posterior_predictive(node.data,
                                                            node.data_uncerts))

        if recurse:
            if node.left_child is not None:
                self.add_node_posterior(node.left_child, GMM,
                                        posterior_preds=posterior_preds,
                                        weight_mult=weight_mult,
                                        recurse=True)

            if node.right_child is not None:
                self.add_node_posterior(node.right_child, GMM,
                                        posterior_preds=posterior_preds,
                                        weight_mult=weight_mult,
                                        recurse=True)

    def get_single_posteriors(self):
        """ get_single_posteriors()

            Find the posteriors for each data point as a Gaussian
            mixture, with each component in he mixture corresponding
            to a node that the data point appears in.

        """
        self.post_GMMs = []

        # get mixture models for each data point

        for it in range(self.Nleaves):
            path = self.find_path(it)

            # initialise a GMM
            post_GMM = gmm.GMM()

            node = self.root_node

            self.add_node_posterior(node, post_GMM, recurse=False)

            for direction in path:
                if direction == "left":
                    node = node.left_child
                elif direction == "right":
                    node = node.right_child

                self.add_node_posterior(node, post_GMM, recurse=False)

            post_GMM.normalise_weights()
            post_GMM.set_mean_covar()
            self.post_GMMs.append(post_GMM)

    def get_cavity_priors(self):
        """ get_cavity_priors()

            Find the 'cavity priors' for each data point as a Gaussian
            mixture, with each component in he mixture corresponding
            to a node that the data point appears in.

        """
        self.cavity_GMMs = []

        # get mixture models for each data point

        for it in range(self.Nleaves):
            path = self.find_path(it)

            # initialise a GMM
            cavity_GMM = gmm.GMM()

            node = self.root_node

            weight = node.prev_wk*math.exp(node.log_rk)
            mu, sigma = node.data_model.cavity_prior(
                            self.data[it], self.data_uncerts[it],
                            node.params)
            cavity_GMM.add_component(weight, mu, sigma)

            for direction in path:
                if direction == "left":
                    node = node.left_child
                elif direction == "right":
                    node = node.right_child

                mu, sigma = node.data_model.cavity_prior(
                                self.data[it], self.data_uncerts[it],
                                node.params)

                if node.log_rk is not None:
                    weight = node.prev_wk*math.exp(node.log_rk)
                else:           # a leaf
                    weight = node.prev_wk

                cavity_GMM.add_component(weight, mu, sigma)

            cavity_GMM.normalise_weights()
            cavity_GMM.set_mean_covar()
            self.cavity_GMMs.append(cavity_GMM)

    def set_params(self, node):

        node.get_node_params()

        if node.left_child is not None:
            child_prev_wk = (node.prev_wk*(1-math.exp(node.log_rk)))
            node.left_child.prev_wk = child_prev_wk
            self.set_params(node.left_child)
        if node.right_child is not None:
            node.right_child.prev_wk = child_prev_wk
            self.set_params(node.right_child)

    def tree_posterior_predictive_prob(self, node, new_data, new_data_uncerts,
                                       target_prob=np.inf):
        """ tree_posterior_predictive_prob(new_data, new_data_uncerts,
                                           target_prob=np.inf)

            Evaluate the posterior predictive probability of some
            data with uncertainties given a BHC clustering tree below
            some node

            Parameters
            ----------
            new_data : ndarray
                The new data
            new_data_uncerts : ndarray
                The uncertainties on the new data
            node : noisy_node
                The root node of the tree whose posterior predicitive
                is being calculated
            target_prob : float
                A probability against which this calculation is being
                compared. Halt calculation when this probability is
                surpassed

            Returns
            -------
            log_pp : float
                The logarithm of the posterior predictive probability
                of the new data
        """
        if node.log_rk is not None:
            weight = math.exp(node.log_rk)
        else:           # a leaf
            weight = 1.

        pp = weight * math.exp(node.posterior_predictive_prob(
                                                new_data, new_data_uncerts))

        if node.left_child is not None and pp <= target_prob:
            pp += ((1-weight)/2.
                   * self.tree_posterior_predictive_prob(node.left_child,
                                                         new_data,
                                                         new_data_uncerts))

        if node.right_child is not None and pp <= target_prob:
            pp += ((1-weight)/2.
                   * self.tree_posterior_predictive_prob(node.right_child,
                                                         new_data,
                                                         new_data_uncerts))

        return pp

    def __str__(self):
        bhc_str = ("==================================\n"
                   "BHC fit to {0} (noisy) data points, with "
                   "alpha={1} .\n".format(self.data.shape[0], self.crp_alpha))

        # start from root node
        l_it = 0
        prev_nodes = {0: self.root_node}

        # Now iterate over levels
        while len(prev_nodes) > 0:
            l_it += 1
            bhc_str += "===== LEVEL {0} =====\n".format(l_it)
            nodes = {}

            for i in sorted(prev_nodes.keys()):
                # If children exist
                if prev_nodes[i].log_rk is not None:
                    bhc_str += ("node : {0} size : {1} ({2}) "
                                "node_prob : {3:.5G} ({4:G} {5:G})\n".format(
                                       i, prev_nodes[i].nk,
                                       prev_nodes[i].data.shape[0],
                                       prev_nodes[i].prev_wk
                                       * math.exp(prev_nodes[i].log_rk),
                                       prev_nodes[i].params[0][0],
                                       prev_nodes[i].params[0][1]))
                    nodes[i*2] = prev_nodes[i].left_child
                    nodes[i*2+1] = prev_nodes[i].right_child

                # if leaf
                else:
                    bhc_str += ("node : {0} size : {1}({2}) "
                                "node_prob : {3:.5G} ({4:G} {5:G})\n".format(
                                       i, prev_nodes[i].nk,
                                       prev_nodes[i].data.shape[0],
                                       prev_nodes[i].prev_wk,
                                       prev_nodes[i].params[0][0],
                                       prev_nodes[i].params[0][1]))

            prev_nodes = nodes

        return bhc_str


class noisy_Node(object):
    """ A node in the hierarchical clustering.
    Attributes
    ----------
    nk : int
        Number of data points assigned to the node
    data : numpy.ndarrary (n, d)
        The data assigned to the Node. Each row is a datum.
    data_model : idsteach.CollapsibleDistribution
        The data model used to calcuate marginal likelihoods
    crp_alpha : float
        Chinese restaurant process concentration parameter
    log_dk : float
        Used in the calculation of the prior probability. Defined in
        Fig 3 of Heller & Ghahramani (2005).
    log_pi : float
        Prior probability that all associated leaves belong to one
        cluster.
    log_ml : float
        The log marginal likelihood for the tree of the node and
        its children. This is given by eqn 2 of Heller &
        Ghahrimani (2005). Note that this definition is
        recursive.  Do not define if the node is
        a leaf.
    logp : float
        The log marginal likelihood for the particular cluster
        represented by the node. Given by eqn 1 of Heller &
        Ghahramani (2005).
    log_rk : float
        The log-probability of the merge that created the node. For
        nodes that are leaves (i.e. not created by a merge) this is
        None.
    left_child : Node
        The left child of a merge. For nodes that are leaves (i.e.
        the original data points and not made by a merge) this is
        None.
    right_child : Node
        The right child of a merge. For nodes that are leaves
        (i.e. the original data points and not made by a merge)
        this is None.
    index : int
        The indexes of the leaves associated with the node in some
        indexing scheme.
    prev_wk : float
        The product of the (1-r_k) factors for the nodes leading
        to this node from (and including) the root node. Used in
        eqn 9 of Heller & ghahramani (2005a).
    """

    def __init__(self, data, data_uncerts, data_model, crp_alpha=1.0,
                 log_dk=None, log_pi=0.0, log_ml=None, logp=None,
                 sum_dict=None, log_rk=None, left_child=None, right_child=None,
                 nk=1, indexes=None):
        """
        Parameters
        ----------
        data : numpy.ndarray
            Array of data_model-appropriate data
        data_model : idsteach.CollapsibleDistribution
            The data model used to calcuate marginal likelihoods
        crp_alpha : float (0, Inf)
            CRP concentration parameter
        log_dk : float
            Cached probability variable. Do not define if the node is
            a leaf.
        log_pi : float
            Cached probability variable. Do not define if the node is
            a leaf.
        log_ml : float
            The log marginal likelihood for the tree of the node and
            its children. This is given by eqn 2 of Heller &
            Ghahrimani (2005). Note that this definition is
            recursive.  Do not define if the node is
            a leaf.
        logp : float
            The log marginal likelihood for the particular cluster
            represented by the node. Given by eqn 1 of Heller &
            Ghahramani (2005).
        sum_dict : dict
            A dictionary of various summations that are used to speed
            up the calculation of logp
        log_rk : float
            The probability of the merged hypothesis for the node.
            Given by eqn 3 of Heller & Ghahrimani (2005). Do not
            define if the node is a leaf.
        left_child : Node, optional
            The left child of a merge. For nodes that are leaves (i.e.
            the original data points and not made by a merge) this is
            None.
        right_child : Node, optional
            The right child of a merge. For nodes that are leaves
            (i.e. the original data points and not made by a merge)
            this is None.
        nk : int, optional
            The number of components that have gone into building the
            cluster. Each component is normally an individual datum.
            But if building from pre-clustered data, the base component
            is instead these pre-clusters
        index : int, optional
            The index of the node in some indexing scheme.
        """
        self.data_model = data_model
        self.data = data
        self.data_uncerts = data_uncerts

        self.nk = nk
        self.crp_alpha = crp_alpha
        self.log_alpha = math.log(crp_alpha)
        self.log_pi = log_pi
        self.log_rk = log_rk

        self.left_child = left_child
        self.right_child = right_child

        if isinstance(indexes, int):
            self.indexes = [indexes]
        else:
            self.indexes = indexes

        if log_dk is None:
            if self.nk == 1:
                self.log_dk = math.log(crp_alpha)

            else:
                # approxation to d_{left}_k * d_{right}_k
                log_d_tree = 0
                for i in range(2, self.nk+1):
                    log_d_tree = np.logaddexp(self.log_alpha + math.lgamma(i),
                                              log_d_tree)

                self.log_dk = np.logaddexp(self.log_alpha
                                           + math.lgamma(self.nk),
                                           log_d_tree)
        else:
            self.log_dk = log_dk

        if logp is None or sum_dict is None:    # i.e. for a leaf
            (self.logp, self.sum_dict) = self.data_model.\
                                         log_marginal_likelihood(
                                                self.data, self.data_uncerts)
        else:
            self.logp = logp
            self.sum_dict = sum_dict

        if log_ml is None:  # i.e. for a leaf
            self.log_ml = self.logp
        else:
            self.log_ml = log_ml
        self.prev_wk = 1.

    @classmethod
    def as_merge(cls, node_left, node_right):
        """ Create a node from two other nodes
        Parameters
        ----------
        node_left : Node
            the Node on the left
        node_right : Node
            The Node on the right
        """
        crp_alpha = node_left.crp_alpha
        data_model = node_left.data_model
        data = np.vstack((node_left.data, node_right.data))
        data_uncerts = np.vstack((node_left.data_uncerts,
                                  node_right.data_uncerts))

        indexes = node_left.indexes + node_right.indexes
        indexes.sort()

        nk = node_left.nk + node_right.nk
        log_dk = logaddexp(math.log(crp_alpha) + math.lgamma(nk),
                           node_left.log_dk + node_right.log_dk)
        log_pi = -math.log1p(math.exp(node_left.log_dk
                                      + node_right.log_dk
                                      - math.log(crp_alpha)
                                      - math.lgamma(nk)))

        # combine sum_dicts

        sum_dict = {}
        for key in node_left.sum_dict:
            if key in node_right.sum_dict:
                sum_dict[key] = (node_left.sum_dict[key]
                                 + node_right.sum_dict[key])

        # Calculate log_rk - the log probability of the merge

        logp, sum_dict = data_model.log_marginal_likelihood(data, data_uncerts,
                                                            **sum_dict)
        numer = log_pi + logp

        neg_pi = math.log(-math.expm1(log_pi))
        log_ml = logaddexp(numer, neg_pi+node_left.log_ml+node_right.log_ml)

        log_rk = numer-log_ml

        if log_pi == 0:
            raise RuntimeError('Precision error')

        return cls(data, data_uncerts, data_model, crp_alpha, log_dk,
                   log_pi, log_ml, logp, sum_dict, log_rk, node_left,
                   node_right, nk, indexes)

    def get_node_params(self):
        self.params = self.data_model.update_parameters(
                                              self.data,
                                              self.data_uncerts,
                                              self.data_model.mu_0,
                                              self.data_model.sigma_0,
                                              self.data_model.S,
                                              self.data_model.d)

    def posterior_predictive_prob(self, new_data, new_data_uncerts):
        """ posterior_predictive_prob(new_data, new_data_uncerts)

            Evaluate the posterior predictive probability of some
            data with uncertainties given the data in the node.

            Parameters
            ----------
            new_data : ndarray
                The new data
            new_data_uncerts : ndarray
                The uncertainties on the new data

            Returns
            -------
            log_pp : float
                The logarithm of the posterior predictive probability
                of the new data
        """
        log_pp = self.data_model.log_posterior_predictive(
                                            new_data, new_data_uncerts,
                                            self.data, self.data_uncerts)
        return log_pp
