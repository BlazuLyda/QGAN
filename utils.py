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