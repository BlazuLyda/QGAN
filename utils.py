import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp

def expval_from_statevector(circ: QuantumCircuit, bind: dict, op: SparsePauliOp) -> float:
    # Bind params
    bound_circ = circ.assign_parameters(bind, inplace=False)
    # Evolve
    sv = Statevector.from_instruction(bound_circ)
    # Expectation
    return float(np.real(sv.expectation_value(op)))

def avg_z_op(total_qubits: int, target_qubit: int):
    """
    Returns Z operator on the target qubit, identity elsewhere.
    Paper: Measure Z on 'Out D' register.
    """
    # Qiskit string order is reversed: q_n ... q_0
    # If target is 0, Z is at the far right.
    s = ["I"] * total_qubits
    s[total_qubits - 1 - target_qubit] = "Z"
    return SparsePauliOp.from_list([("".join(s), 1.0)])

def pauli_op(total_qubits: int, pauli_map: dict[int, str]):
    """
    Build a SparsePauliOp for arbitrary Pauli observables.

    pauli_map: {qubit_index: "X" | "Y" | "Z"}
    """
    s = ["I"] * total_qubits
    for q, p in pauli_map.items():
        s[total_qubits - 1 - q] = p
    return SparsePauliOp.from_list([("".join(s), 1.0)])


def double_qubit_op(
    total_qubits: int,
    decision_qubit: int = 0,
    other_qubit: int = 1,
    lambda_ZZ: float = 0.05,
    lambda_XX: float = 0.05,
):
    """
    Builds an aggregated Pauli observable and returns an evaluator function.

    Observable:
        O = Z_decision
          + lambda_ZZ * Z_decision Z_data
          + lambda_XX * X_decision X_data

    Returns:
        evaluator(statevector) -> float
    """

    # Build individual Pauli operators
    Z0 = pauli_op(
        total_qubits,
        {decision_qubit: "Z"}
    )
    ZZ = pauli_op(
        total_qubits,
        {decision_qubit: "Z", other_qubit: "Z"}
    )
    XX = pauli_op(
        total_qubits,
        {decision_qubit: "X", other_qubit: "X"},
    )

    # Aggregate into a single observable
    observable = Z0 + lambda_ZZ * ZZ + lambda_XX * XX
    return observable

