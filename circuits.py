import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from utils import pauli_string_on_qubit, avg_local_xyz_op


class QGANCircuits:
    def __init__(self, n_qubits=3, n_layers=2):
        """
        :param n_qubits: Number of qubits for data/generator.
        :param n_layers: Depth of the ansatz (per circuit).
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # We'll use an ansatz with 3 params per qubit per layer (RX, RY, RZ)
        self.params_per_qubit = 3

        # Calculate total parameters needed
        self.num_params = self.n_qubits * self.n_layers * self.params_per_qubit

        # Define Parameter Vectors
        self.gen_params = ParameterVector("g", self.num_params)
        self.disc_params = ParameterVector("d", self.num_params)

        # Real data is also an arbitrary circuit of the same size,
        # just with fixed random angles.
        self.real_params = ParameterVector("r", self.num_params)

        # Pre-build Circuits
        self.measure_op = avg_local_xyz_op(n_qubits)

        self.real_disc_circuit = self._build_real_disc()
        self.gen_disc_circuit = self._build_gen_disc()

    def _add_scalable_ansatz(self, circ, params):
        """
        Applies a scalable layer of Rotations followed by CNOT ring.
        """
        param_idx = 0

        for layer in range(self.n_layers):
            # 1. Rotations on all qubits
            for q in range(self.n_qubits):
                # Apply 3 rotations (Universal single qubit gate)
                circ.rx(params[param_idx], q)
                circ.ry(params[param_idx+1], q)
                circ.rz(params[param_idx+2], q)
                param_idx += self.params_per_qubit

            # 2. Entanglement (Ring topology)
            if self.n_qubits > 1:
                for q in range(self.n_qubits):
                    # Connect q to q+1 (wrapping around)
                    target = (q + 1) % self.n_qubits
                    circ.cx(q, target)

    def _build_real_disc(self):
        circ = QuantumCircuit(self.n_qubits)
        self._add_scalable_ansatz(circ, self.real_params)   # Real Data
        self._add_scalable_ansatz(circ, self.disc_params)   # Discriminator
        return circ

    def _build_gen_disc(self):
        circ = QuantumCircuit(self.n_qubits)
        self._add_scalable_ansatz(circ, self.gen_params)    # Generator
        self._add_scalable_ansatz(circ, self.disc_params)   # Discriminator
        return circ

    def get_bloch_vector_real(self, real_w_values):
        circ = QuantumCircuit(self.n_qubits)
        self._add_scalable_ansatz(circ, self.real_params)
        return self._get_avg_bloch(circ, real_w_values)

    def get_bloch_vector_generator(self, gen_w_values):
        circ = QuantumCircuit(self.n_qubits)
        self._add_scalable_ansatz(circ, self.gen_params)
        return self._get_avg_bloch(circ, gen_w_values)

    def _get_avg_bloch(self, circ, param_values_list):
        from qiskit.quantum_info import Statevector

        params_in_circ = list(circ.parameters)
        bind = {params_in_circ[i]: param_values_list[i] for i in range(len(params_in_circ))}

        sv = Statevector.from_instruction(circ.assign_parameters(bind))

        blochs = []
        for q in range(self.n_qubits):
            Xq = pauli_string_on_qubit("X", q, self.n_qubits)
            Yq = pauli_string_on_qubit("Y", q, self.n_qubits)
            Zq = pauli_string_on_qubit("Z", q, self.n_qubits)

            blochs.append([
                np.real(sv.expectation_value(Xq)),
                np.real(sv.expectation_value(Yq)),
                np.real(sv.expectation_value(Zq))
            ])

        return np.mean(blochs, axis=0)

