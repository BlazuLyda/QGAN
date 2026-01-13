import numpy as np
from qiskit.quantum_info import Statevector
from typing import List


class QuantumEnsemble:
    """
    Represents a probabilistic ensemble of quantum states.
    """

    def __init__(self, states: List[Statevector], probs: List[float]):
        assert len(states) == len(probs), "States and probabilities must match"
        assert np.isclose(sum(probs), 1.0), "Probabilities must sum to 1"

        self.states = states
        self.probs = np.array(probs)

    def sample_statevector(self) -> Statevector:
        """Sample a pure state according to the ensemble distribution."""
        idx = np.random.choice(len(self.states), p=self.probs)
        return self.states[idx]


class QuantumDataSource:
    """
    Multi-class quantum data source.
    """

    def __init__(self, ensembles: list[QuantumEnsemble]):
        self.ensembles = ensembles
        self.num_classes = len(ensembles)

    def sample_class(self, lam: int):
        """
        Sample from class `lam`.

        Args:
            lam: class index
            return_density: if True, return density matrix instead of pure state
        """
        assert 0 <= lam < self.num_classes, "Invalid class index"

        ensemble = self.ensembles[lam]
        return ensemble.sample_statevector()



