import numpy as np
import pytest
import matplotlib.pyplot as plt
import logging
import sys
from sklearn.datasets import make_blobs
from pathlib import Path
from inspect import stack

from pyBHC.bhc import bhc
from pyBHC.dists import NormalInverseWishart


TEST_OUTPUT_PATH = Path('tests/output')

# LOGGING_LEVEL = logging.INFO
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
        savefig_path = TEST_OUTPUT_PATH / \
            stack()[1].function / ("n_" + str(len(data)))
        savefig_path.parent.mkdir(parents=True, exist_ok=True)

        bhc_model.plot_dendrogram()
        # plt.show()
        plt.savefig(savefig_path.with_name(
            savefig_path.name+'_dendrogram.png'), format='png')
        plt.close()
        bhc_model.plot_clusters()
        if len(data) > 1:
            true_clusters = [list(range(len(data)))]
            plt.title("Purity: {}".format(
                bhc_model.compute_dendrogram_purity(true_clusters)))
        plt.savefig(savefig_path.with_name(
            savefig_path.name+'_clusters.png'), format='png')
        # plt.show()
        plt.close()

    k, p = bhc_model.predict(data[-1])

    print(bhc_model.iteration_run_times)

    return bhc_model


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
        savefig_path = TEST_OUTPUT_PATH / \
            stack()[1].function / ("n_" + str(len(data)))
        savefig_path.parent.mkdir(parents=True, exist_ok=True)

        bhc_model.plot_dendrogram()
        plt.savefig(savefig_path.with_name(
            savefig_path.name+'_dendrogram.png'), format='png')
        # plt.show()
        plt.close()
        bhc_model.plot_clusters()
        if len(data) > 1:
            true_clusters = [list(range(len(data)))]
            plt.title("Purity: {}".format(
                bhc_model.compute_dendrogram_purity(true_clusters)))

        # TODO: save plot based on test name
        plt.savefig(savefig_path.with_name(
            savefig_path.name+'_clusters.png'), format='png')
        # plt.show()
        plt.close()

    k, p = bhc_model.predict(data[-1])

    print(bhc_model.iteration_run_times)


def generate_test_data(n_samples):
    X, y = make_blobs(n_samples=n_samples, centers=3,
                      n_features=2,  random_state=0,
                      cluster_std=0.2)
    return list(X)
    # return list(np.random.normal(size=(n_samples, 2)))


# @pytest.mark.parametrize("n_samples", [1, 2, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75])
@pytest.mark.parametrize("n_samples", [50, 55, 60, 65, 70, 75])
def test_bhc_w_no_plot(n_samples):
    data = generate_test_data(n_samples)
    train_model(data, False)


# @pytest.mark.parametrize("n_samples", [1, 2, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75])
@pytest.mark.parametrize("n_samples", [50, 55, 60, 65, 70, 75])
def test_random_bhc_w_no_plot(n_samples):
    data = generate_test_data(n_samples)
    train_model_randomized(data, False)


# @pytest.mark.parametrize("n_samples", [1, 2, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75])
# @pytest.mark.parametrize("n_samples", [10])
@pytest.mark.parametrize("n_samples", [50, 55, 60, 65, 70, 75])
def test_bhc_w_plot(n_samples):
    data = generate_test_data(n_samples)
    train_model(data, True)


# @pytest.mark.parametrize("n_samples", [1, 2, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75])
@pytest.mark.parametrize("n_samples", [50, 55, 60, 65, 70, 75])
def test_random_bhc_w_plot(n_samples):
    data = generate_test_data(n_samples)
    train_model_randomized(data, True)


# wrong assumption
# @pytest.mark.parametrize("n_samples", [10])
# def test_data_id_order(n_samples):
#    data = generate_test_data(n_samples)
#    bhc_model = train_model(data, False)
#    ids = bhc_model.root_node.pre_order(lambda x: x.id)
#    assert all(ids[i] < ids[i+1] for i in range(len(ids)-1)), \
#        'ids not sorted correctly; {}'.format(ids)
