import numpy as np
from typing import Any
from qiskit.quantum_info import DensityMatrix
import torch

from circuits import GenCircuit 
from data import QuantumDataSource, QuantumEnsemble
from gradients import tensor_to_bind_dict


def estimate_generator_density(
    generator_sampler,
    n_gen_samples: int,
) -> DensityMatrix:
    """
    Monte Carlo estimate of the generator's expected density matrix.
    """
    rho_sum = None

    for _ in range(n_gen_samples):
        rho = generator_sampler().data
        rho_sum = rho if rho_sum is None else rho_sum + rho

    rho_bar = rho_sum / np.trace(rho_sum) if rho_sum is not None else np.zeros_like(rho_sum)
    return DensityMatrix(rho_bar)


def ensemble_cross_entropy_density(
    ensemble: QuantumEnsemble,
    rho_gen: DensityMatrix,
    eps: float = 1e-12,
) -> float:
    """
    Compute H(p_real, p_G) using density matrices.
    """
    H = 0.0

    for psi, p_real in zip(ensemble.states, ensemble.probs):
        ket = psi.data.reshape(-1, 1)
        # Density matrix of the state from ensemble |psi><psi|
        proj = ket @ ket.conj().T

        # Probability assigned by generator to this state
        p_g = np.real(np.trace(rho_gen.data @ proj))
        p_g = max(p_g, eps)

        # Cross-entropy contribution
        H -= p_real * np.log(p_g)

    return float(H)

def compute_cross_entropy_over_labels_tensor(
    data_source: QuantumDataSource,
    gen: GenCircuit,
    gen_w: torch.Tensor,
    n_gen_samples_per_label: int = 2000,
    eps: float = 1e-12,
) -> list[float]:

    return compute_cross_entropy_over_labels(
        data_source,
        gen,
        tensor_to_bind_dict(gen_w, gen.gen_params),
        n_gen_samples_per_label,
        eps,
    )


def compute_cross_entropy_over_labels(
    data_source: QuantumDataSource,
    gen: GenCircuit,
    param_bind: dict[Any, float],
    n_gen_samples_per_label: int = 2000,
    eps: float = 1e-12,
) -> list[float]:

    cross_entropies = []

    for label in range(data_source.num_classes):

        # Real data
        ensemble = data_source.get_ensemble(label)

        # Generator density matrix (Monte Carlo)
        def generator_sampler():
            return gen.sample_data_density_matrix(
                    label, param_bind
                )
        rho_gen = estimate_generator_density(
            generator_sampler,
            n_gen_samples_per_label,
        )

        H = ensemble_cross_entropy_density(
            ensemble,
            rho_gen,
            eps=eps,
        )
        cross_entropies.append(H)

    return cross_entropies


