import numpy as np
import pytest
import matplotlib.pyplot as plt
import logging
import sys


from pyBHC.bhc import bhc
from pyBHC.dists import NormalInverseWishart

#LOGGING_LEVEL = logging.INFO
LOGGING_LEVEL = logging.DEBUG

logging.basicConfig(
    level=LOGGING_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        # logging.FileHandler(LOGPATH/'{}.log'.format(NOW)),
        logging.StreamHandler(sys.stdout)
    ])

logger = logging.getLogger(__name__)

hypers = {
    'mu_0': np.zeros(2),
    'nu_0': 3.0,
    'kappa_0': 1.0,
    'lambda_0': np.eye(2),
}

crp_alpha = 2.0


def train_model(data, plot_output):
    bhc_model = bhc(data, NormalInverseWishart(**hypers), crp_alpha=crp_alpha)
    bhc_model.fit()

    # Verify rks
    rks = np.array(bhc_model.rks)
    assert np.all(np.logical_and(rks > 0, rks < 1))

    # Verify omegas
    assert len(bhc_model.omegas) == len(bhc_model.nodes)
    omega_sum = np.sum(np.exp(list(bhc_model.omegas.values())))
    # print(len(bhc_model.omegas))
    # print(np.exp(bhc_model.omegas))
    print(omega_sum)
    assert np.isclose(omega_sum, 1)

    print(str(bhc_model.root_node))

    if plot_output:
        bhc_model.plot_dendrogram()
        plt.show()
        plt.close()
        plot_data = np.array(data)
        bhc_model.plot_clusters(data=plot_data)
        if len(data) > 1:
            true_clusters = [list(range(len(data)))]
            plt.title("Purity: {}".format(
                bhc_model.compute_dendrogram_purity(true_clusters)))
        plt.show()
        plt.close()

    k, p = bhc_model.predict(data[-1])


def train_model_randomized(data, plot_output):
    bhc_model = bhc(data, NormalInverseWishart(**hypers), crp_alpha=crp_alpha)
    bhc_model.randomized_fit()

    # Verify rks
    rks = np.array(bhc_model.rks)
    assert np.all(np.logical_and(rks > 0, rks <= 1))

    # Verify omegas
    assert len(bhc_model.omegas) == len(bhc_model.nodes)
    omega_sum = np.sum(np.exp(list(bhc_model.omegas.values())))
    # print(len(bhc_model.omegas))
    # print(np.exp(bhc_model.omegas))
    print(omega_sum)
    assert np.isclose(omega_sum, 1)

    print(str(bhc_model.root_node))

    if plot_output:
        bhc_model.plot_dendrogram()
        plt.show()
        plt.close()
        plot_data = np.array(data)
        bhc_model.plot_clusters(data=plot_data)
        if len(data) > 1:
            true_clusters = [list(range(len(data)))]
            plt.title("Purity: {}".format(
                bhc_model.compute_dendrogram_purity(true_clusters)))
        plt.show()
        plt.close()

    k, p = bhc_model.predict(data[-1])


def test_bhc_1_w_plot():
    data = list(np.random.normal(size=(1, 2)))
    train_model(data, True)


def test_bhc_2_w_plot():
    data = list(np.random.normal(size=(2, 2)))
    train_model(data, True)


def test_bhc_3_w_plot():
    data = list(np.random.normal(size=(3, 2)))
    train_model(data, True)


def test_bhc_4_w_plot():
    data = list(np.random.normal(size=(4, 2)))
    train_model(data, True)


def test_bhc_5_w_plot():
    data = list(np.random.normal(size=(5, 2)))
    train_model(data, True)


def test_bhc_6_w_plot():
    data = list(np.random.normal(size=(6, 2)))
    train_model(data, True)


def test_bhc_7_w_plot():
    data = list(np.random.normal(size=(7, 2)))
    train_model(data, True)


def test_bhc_8_w_plot():
    data = list(np.random.normal(size=(8, 2)))
    train_model(data, True)


def test_bhc_9_w_plot():
    data = list(np.random.normal(size=(9, 2)))
    train_model(data, True)


def test_bhc_25_w_plot():
    data = list(np.random.normal(size=(25, 2)))
    train_model(data, True)


def test_bhc_50_w_plot():
    data = list(np.random.normal(size=(50, 2)))
    train_model(data, True)


def test_random_bhc_1_w_plot():
    data = list(np.random.normal(size=(1, 2)))
    train_model_randomized(data, True)


def test_random_bhc_2_w_plot():
    data = list(np.random.normal(size=(2, 2)))
    train_model_randomized(data, True)


def test_random_bhc_3_w_plot():
    data = list(np.random.normal(size=(3, 2)))
    train_model_randomized(data, True)


def test_random_bhc_4_w_plot():
    data = list(np.random.normal(size=(4, 2)))
    train_model_randomized(data, True)


def test_random_bhc_5_w_plot():
    data = list(np.random.normal(size=(5, 2)))
    train_model_randomized(data, True)


def test_random_bhc_6_w_plot():
    data = list(np.random.normal(size=(6, 2)))
    train_model_randomized(data, True)


def test_random_bhc_7_w_plot():
    data = list(np.random.normal(size=(7, 2)))
    train_model_randomized(data, True)


def test_random_bhc_8_w_plot():
    data = list(np.random.normal(size=(8, 2)))
    train_model_randomized(data, True)


def test_random_bhc_9_w_plot():
    data = list(np.random.normal(size=(9, 2)))
    train_model_randomized(data, True)


def test_random_bhc_25_w_plot():
    data = list(np.random.normal(size=(25, 2)))
    train_model_randomized(data, True)


def test_random_bhc_50_w_plot():
    data = list(np.random.normal(size=(50, 2)))
    train_model_randomized(data, True)


def test_bhc_1_w_no_plot():
    data = list(np.random.normal(size=(1, 2)))
    train_model(data, False)


def test_bhc_2_w_no_plot():
    data = list(np.random.normal(size=(2, 2)))
    train_model(data, False)


def test_bhc_3_w_no_plot():
    data = list(np.random.normal(size=(3, 2)))
    train_model(data, False)


def test_bhc_4_w_no_plot():
    data = list(np.random.normal(size=(4, 2)))
    train_model(data, False)


def test_bhc_5_w_no_plot():
    data = list(np.random.normal(size=(5, 2)))
    train_model(data, False)


def test_bhc_6_w_no_plot():
    data = list(np.random.normal(size=(6, 2)))
    train_model(data, False)


def test_bhc_7_w_no_plot():
    data = list(np.random.normal(size=(7, 2)))
    train_model(data, False)


def test_bhc_8_w_no_plot():
    data = list(np.random.normal(size=(8, 2)))
    train_model(data, False)


def test_bhc_9_w_no_plot():
    data = list(np.random.normal(size=(9, 2)))
    train_model(data, False)


def test_bhc_25_w_no_plot():
    data = list(np.random.normal(size=(25, 2)))
    train_model(data, False)


def test_bhc_50_w_no_plot():
    data = list(np.random.normal(size=(50, 2)))
    train_model(data, False)


def test_random_bhc_1_w_no_plot():
    data = list(np.random.normal(size=(1, 2)))
    train_model_randomized(data, False)


def test_random_bhc_2_w_no_plot():
    data = list(np.random.normal(size=(2, 2)))
    train_model_randomized(data, False)


def test_random_bhc_3_w_no_plot():
    data = list(np.random.normal(size=(3, 2)))
    train_model_randomized(data, False)


def test_random_bhc_4_w_no_plot():
    data = list(np.random.normal(size=(4, 2)))
    train_model_randomized(data, False)


def test_random_bhc_5_w_no_plot():
    data = list(np.random.normal(size=(5, 2)))
    train_model_randomized(data, False)


def test_random_bhc_6_w_no_plot():
    data = list(np.random.normal(size=(6, 2)))
    train_model_randomized(data, False)


def test_random_bhc_7_w_no_plot():
    data = list(np.random.normal(size=(7, 2)))
    train_model_randomized(data, False)


def test_random_bhc_8_w_no_plot():
    data = list(np.random.normal(size=(8, 2)))
    train_model_randomized(data, False)


def test_random_bhc_9_w_no_plot():
    data = list(np.random.normal(size=(9, 2)))
    train_model_randomized(data, False)


def test_random_bhc_25_w_no_plot():
    data = list(np.random.normal(size=(25, 2)))
    train_model_randomized(data, False)


def test_random_bhc_50_w_no_plot():
    data = list(np.random.normal(size=(50, 2)))
    train_model_randomized(data, False)
