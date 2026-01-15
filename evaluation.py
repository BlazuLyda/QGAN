import numpy as np
from qiskit.quantum_info import Statevector

from data import QuantumEnsemble


def fidelity(psi: Statevector, phi: Statevector) -> float:
    """
    Computes similarity between two quantum states via Fidelity. 
    """
    return np.abs(np.vdot(psi.data, phi.data)) ** 2


def estimate_p_g(
    gen_samples: list[Statevector],
    target_state: Statevector,
) -> float:
    """
    Estimate p_G(|target_state>) via Monte Carlo sampling.
    This gives a measure of how likely G outputs a state similar
    to the target state.
    """
    fidelities = [
        np.abs(np.vdot(target_state.data, phi.data)) ** 2
        for phi in gen_samples
    ]
    return float(np.mean(fidelities))


def sample_cross_entropy(
    ensemble: QuantumEnsemble,
    generator_sampler,
    n_gen_samples: int = 2000,
    eps: float = 1e-12,
) -> float:
    """
    Monte Carlo estimate of cross entropy H(p_real, p_G).
    """
    # Start by generating samples from the learned model (generator),
    # which give an approximation of the probabilities G assigns
    # to the Hilbert space (space of quantum states).
    gen_samples = [generator_sampler() for _ in range(n_gen_samples)]

    # Now compute the sample cross entropy by iterating over all of
    # the states belonging to the real data and comparing their true
    # probability of occuring to the one given by the Generator.
    H = 0.0
    for psi, p_real in zip(ensemble.states, ensemble.probs):
        p_g = estimate_p_g(gen_samples, psi)
        p_g = max(p_g, eps)
        H -= p_real * np.log(p_g)

    return H

