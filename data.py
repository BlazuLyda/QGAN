import numpy as np
from qiskit.quantum_info import Statevector

class QuantumEnsemble:
    """
    Represents a probabilistic ensemble of quantum states.
    """

    def __init__(self, states: list[Statevector], probs: list[float]):
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
    def __init__(self, n_qubits_data: int, n_qubits_label: int, ensembles: list[QuantumEnsemble]):

        self.n_data = n_qubits_data
        self.n_label = n_qubits_label
        self.ensembles = ensembles
        self.num_classes = len(ensembles)

        if (2 ** n_qubits_label) < self.num_classes:
            raise ValueError("Not enough label qubits to represent all classes.")

    def sample_class(self, lam: int) -> Statevector:
        """
        Sample from class `lam`.

        Args:
            lam: class index
        """
        assert 0 <= lam < self.num_classes, "Invalid class index"

        ensemble = self.ensembles[lam]
        return ensemble.sample_statevector()
