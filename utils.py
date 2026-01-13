import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Pauli

def pauli_string_on_qubit(op: str, qubit: int, n_qubits: int) -> Pauli:
    """
    Qiskit Pauli string convention: leftmost char is qubit n-1.
    """
    s = ["I"] * n_qubits
    s[n_qubits - 1 - qubit] = op
    return Pauli("".join(s))

def expval_from_statevector(circ: QuantumCircuit, bind: dict, pauli: Pauli) -> float:
    """
    Executes a circuit with bound parameters and calculates expectation value.
    """
    # Create a copy with assigned parameters to avoid modifying the original template
    bound_circ = circ.assign_parameters(bind, inplace=False)
    sv = Statevector.from_instruction(bound_circ)
    return float(np.real(sv.expectation_value(pauli)))

def prob_from_expval(expval: float) -> float:
    """Map Z expectation in [-1,1] to probability in [0,1]."""
    return (expval + 1.0) / 2.0