import numpy as np
import pytest


from pyBHC.bhc import bhc
from pyBHC.dists import NormalInverseWishart


def test_bhc():
    hypers = {
        'mu_0': np.zeros(2),
        'nu_0': 3.0,
        'kappa_0': 1.0,
        'lambda_0': np.eye(2),
    }

    crp_alpha = 1.0

    data = list(np.random.normal(size=(20, 2)))
    bhc_model = bhc(data, NormalInverseWishart(**hypers), crp_alpha=crp_alpha)
    bhc_model.fit()
