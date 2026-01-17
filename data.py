import numpy as np
from qiskit.quantum_info import Statevector, DensityMatrix

class QuantumEnsemble:
    """
    Represents a probabilistic ensemble of quantum states.
    """

    def __init__(self, states: list[Statevector], probs: list[float]):
        assert len(states) == len(probs), "States and probabilities must match"
        assert np.isclose(sum(probs), 1.0), "Probabilities must sum to 1"

        self.states = states
        self.probs = np.array(probs)

    def sample_statevector(self, rng: np.random.Generator) -> Statevector:
        """Sample a pure state according to the ensemble distribution."""
        idx = rng.choice(len(self.states), p=self.probs)
        return self.states[idx]

    def get_density_matrix(self) -> DensityMatrix:
        """
        Return the density matrix corresponding to the ensemble:
        rho = sum_i p_i |psi_i><psi_i|
        """
        dim = self.states[0].dim
        rho = np.zeros((dim, dim), dtype=complex)

        for psi, p in zip(self.states, self.probs):
            ket = psi.data.reshape(-1, 1)
            rho += p * (ket @ ket.conj().T)

        return DensityMatrix(rho)


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

    def sample_class(self, lam: int, rng: np.random.Generator) -> Statevector:
        """
        Sample from class `lam`.

        Args:
            lam: class index
        """
        assert 0 <= lam < self.num_classes, "Invalid class index"

        ensemble = self.ensembles[lam]
        return ensemble.sample_statevector(rng)

    def get_class_density_matrix(self, lam: int) -> DensityMatrix:
        """
        Return the density matrix of class `lam`.
        """
        assert 0 <= lam < self.num_classes, "Invalid class index"
        return self.ensembles[lam].get_density_matrix()

    def get_ensemble(self, label: int) -> QuantumEnsemble:
        """
        Get the ensemble corresponding to a specific label/class.

        Args:
            label: class index
        """
        assert 0 <= label < self.num_classes, "Invalid class index"
        return self.ensembles[label]


def create_cx_data_source() -> QuantumDataSource:
    """
    Creates a simple quantum data source based on the QGAN paper's example.
    """

    # Class 0: |0>
    ket0 = Statevector.from_label('0')
    ensemble0 = QuantumEnsemble(states=[ket0], probs=[1.0])

    # Class 1: |1>
    ket1 = Statevector.from_label('1')
    ensemble1 = QuantumEnsemble(states=[ket1], probs=[1.0])

    return QuantumDataSource(
        n_qubits_data=1,
        n_qubits_label=1,
        ensembles=[ensemble0, ensemble1]
    )
