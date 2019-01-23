import numpy as np
import pytest
import matplotlib.pyplot as plt


from pyBHC.bhc import bhc
from pyBHC.dists import NormalInverseWishart

hypers = {
    'mu_0': np.zeros(2),
    'nu_0': 3.0,
    'kappa_0': 1.0,
    'lambda_0': np.eye(2),
}

crp_alpha = 2.0


def train_model(data):
    bhc_model = bhc(data, NormalInverseWishart(**hypers), crp_alpha=crp_alpha)
    bhc_model.fit()

    # Verify rks
    rks = np.array(bhc_model.rks)
    assert np.all(np.logical_and(rks > 0, rks < 1))

    # Verify omegas
    assert len(bhc_model.omegas) == len(bhc_model.nodes)
    omega_sum = np.sum(np.exp(bhc_model.omegas))
    # print(len(bhc_model.omegas))
    # print(np.exp(bhc_model.omegas))
    print(omega_sum)
    assert np.isclose(omega_sum, 1)

    print(str(bhc_model.root_node))

    bhc_model.plot_dendrogram()
    # plt.show()
    plt.close()
    plot_data = np.array(data)
    bhc_model.plot_clusters(data=plot_data)
    # plt.show()
    plt.close()

    bhc_model.predict(data[-1])


def test_bhc_1():
    data = list(np.random.normal(size=(1, 2)))
    train_model(data)


def test_bhc_2():
    data = list(np.random.normal(size=(2, 2)))
    train_model(data)


def test_bhc_3():
    data = list(np.random.normal(size=(3, 2)))
    train_model(data)


def test_bhc_4():
    data = list(np.random.normal(size=(4, 2)))
    train_model(data)


def test_bhc_5():
    data = list(np.random.normal(size=(5, 2)))
    train_model(data)


def test_bhc_6():
    data = list(np.random.normal(size=(6, 2)))
    train_model(data)


def test_bhc_7():
    data = list(np.random.normal(size=(7, 2)))
    train_model(data)


def test_bhc_8():
    data = list(np.random.normal(size=(8, 2)))
    train_model(data)


def test_bhc_9():
    data = list(np.random.normal(size=(9, 2)))
    train_model(data)


def test_bhc_25():
    data = list(np.random.normal(size=(25, 2)))
    train_model(data)


def test_bhc_50():
    data = list(np.random.normal(size=(50, 2)))
    train_model(data)
