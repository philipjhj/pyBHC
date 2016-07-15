from __future__ import print_function, division
import numpy as np

from scipy.special import multigammaln
from numpy.linalg import slogdet
import math

LOG2PI = math.log(2*math.pi)
LOG2 = math.log(2)


class CollapsibleDistribution(object):
    """ Abstract base class for a family of conjugate distributions.
    """

    def log_marginal_likelihood(self, X):
        """ Log of the marginal likelihood, P(X|prior).  """
        pass


class NormalInverseWishart(CollapsibleDistribution):
    """
    Multivariate Normal likelihood with multivariate Normal prior on 
    mean and Inverse-Wishart prior on the covariance matrix.
    All math taken from Kevin Murphy's 2007 technical report, 
    'Conjugate Bayesian analysis of the Gaussian distribution'.
    """

    def __init__(self, **prior_hyperparameters):
        self.nu_0 = prior_hyperparameters['nu_0']
        self.mu_0 = prior_hyperparameters['mu_0']
        self.kappa_0 = prior_hyperparameters['kappa_0']
        self.lambda_0 = prior_hyperparameters['lambda_0']

        self.d = float(len(self.mu_0))

        self.log_z = self.calc_log_z(self.mu_0, self.lambda_0, self.kappa_0,
                                     self.nu_0)

    @staticmethod
    def update_parameters(X, _mu, _lambda, _kappa, _nu, _d):
        n = X.shape[0]
        xbar = np.mean(X, axis=0)
        kappa_n = _kappa + n
        nu_n = _nu + n
        mu_n = (_kappa*_mu + n*xbar)/kappa_n

        S = np.zeros(_lambda.shape) if n == 1 else (n-1)*np.cov(X.T)
        dt = (xbar-_mu)[np.newaxis]

        back = np.dot(dt.T, dt)
        lambda_n = _lambda + S + (_kappa*n/kappa_n)*back

        assert(mu_n.shape[0] == _mu.shape[0])
        assert(lambda_n.shape[0] == _lambda.shape[0])
        assert(lambda_n.shape[1] == _lambda.shape[1])

        return mu_n, lambda_n, kappa_n, nu_n

    @staticmethod
    def calc_log_z(_mu, _lambda, _kappa, _nu):
        d = len(_mu)
        sign, detr = slogdet(_lambda)
        log_z = (LOG2*(_nu*d/2.0) 
                 + (d/2.0)*math.log(2*math.pi/_kappa) 
                 + multigammaln(_nu/2, d) - (_nu/2.0)*detr)

        return log_z

    def log_marginal_likelihood(self, X):
        n = X.shape[0]
        params_n = self.update_parameters(X, self.mu_0, self.lambda_0,
                                          self.kappa_0, self.nu_0, self.d)
        log_z_n = self.calc_log_z(*params_n)

        return log_z_n - self.log_z - LOG2PI*(n*self.d/2)


class NormalFixedCovar(CollapsibleDistribution):
    """
    Multivariate Normal likelihood with multivariate Normal prior on 
    mean and a fixed covariance matrix.
    All math taken from Kevin Murphy's 2007 technical report, 
    'Conjugate Bayesian analysis of the Gaussian distribution'.
    """

    def __init__(self, **prior_hyperparameters):
        self.mu_0 = prior_hyperparameters['mu_0']
        self.sigma_0 = prior_hyperparameters['sigma_0']
        self.S = prior_hyperparameters['S']

        self.d = float(len(self.mu_0))

        sgn, self.sigma_0_det = np.linalg.slogdet(self.sigma_0)
        sgn, self.S_det = np.linalg.slogdet(self.S)

        self.sigma_0_inv = np.linalg.inv(self.sigma_0)
        self.S_inv = np.linalg.inv(self.S)

        self.log_z0 = self.calc_log_z(self.mu_0, self.sigma_0, self.S)

    @staticmethod
    def update_parameters(X, _mu, _sigma, S, _d):
        n = X.shape[0]
        xbar = np.mean(X, axis=0)

        _sigma_inv = np.linalg.inv(_sigma)
        S_inv = np.linalg.inv(S)

        # update variance on mean
        sigma_n_inv = _sigma_inv + n*S_inv
        sigma_n = np.linalg.inv(sigma_n_inv)

        # update mean
        mu_n = (np.dot(_sigma_inv, _mu)
                + n*np.dot(S_inv, xbar))
        mu_n = np.dot(sigma_n, mu_n)

        assert(mu_n.shape[0] == _mu.shape[0])
        assert(sigma_n.shape[0] == _sigma.shape[0])
        assert(sigma_n.shape[1] == _sigma.shape[1])

        return mu_n, sigma_n, S

    @staticmethod
    def calc_log_z(_mu, _sigma, S):
        d = len(_mu)
        sign, detr = slogdet(_sigma)
        _sigma_inv = np.linalg.inv(_sigma)

        log_z = detr/2 + np.sum(_mu*np.dot(_sigma_inv, _mu))

        return log_z

    def log_marginal_likelihood(self, X):
        n = X.shape[0]
        params_n = self.update_parameters(X, self.mu_0, self.sigma_0,
                                          self.S, self.d)
        log_z_n = self.calc_log_z(*params_n)

        Q = 0.
        for i in range(n):
            Q += np.sum(X[i,:]*np.dot(self.S_inv, X[i,:]))

        lml = (log_z_n - self.log_z0 - LOG2PI*(n*self.d/2) - Q
                - self.S_det*n/2)

        return (log_z_n - self.log_z0 - LOG2PI*(n*self.d/2) - Q
                - self.S_det*n/2)
