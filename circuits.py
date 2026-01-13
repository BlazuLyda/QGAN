import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from utils import pauli_string_on_qubit

class QGANCircuits:
    def __init__(self, n_qubits=3):
        self.n_qubits = n_qubits

        # Define Parameter Vectors
        self.disc_params = ParameterVector("d", 9)
        self.gen_params = ParameterVector("g", 9)

        # Unpack the 3 parameters for real data: phi, theta, omega
        # These are individual Parameter objects, not lists.
        self.phi_p, self.theta_p, self.omega_p = ParameterVector("r", 3)

        # Pre-build the main training circuits
        self.real_disc_circuit = self._build_real_disc()
        self.gen_disc_circuit = self._build_gen_disc()

        # Define the measurement operator (Z on qubit 2)
        self.measure_op = pauli_string_on_qubit("Z", qubit=2, n_qubits=n_qubits)

    @staticmethod
    def _add_real_data(circ, phi, theta, omega):
        circ.h(0)
        circ.rz(phi, 0)
        circ.ry(theta, 0)
        circ.rz(omega, 0)

    @staticmethod
    def _add_generator(circ, w):
        # w is expected to be a list/vector of 9 parameters
        circ.h(0)
        circ.rx(w[0], 0)
        circ.rx(w[1], 1)
        circ.ry(w[2], 0)
        circ.ry(w[3], 1)
        circ.rz(w[4], 0)
        circ.rz(w[5], 1)
        circ.cx(0, 1)
        circ.rx(w[6], 0)
        circ.ry(w[7], 0)
        circ.rz(w[8], 0)

    @staticmethod
    def _add_discriminator(circ, w):
        circ.h(0)
        circ.rx(w[0], 0)
        circ.rx(w[1], 2)
        circ.ry(w[2], 0)
        circ.ry(w[3], 2)
        circ.rz(w[4], 0)
        circ.rz(w[5], 2)
        circ.cx(0, 2)
        circ.rx(w[6], 2)
        circ.ry(w[7], 2)
        circ.rz(w[8], 2)

    def _build_real_disc(self):
        circ = QuantumCircuit(self.n_qubits)
        self._add_real_data(circ, self.phi_p, self.theta_p, self.omega_p)
        self._add_discriminator(circ, self.disc_params)
        return circ

    def _build_gen_disc(self):
        circ = QuantumCircuit(self.n_qubits)
        self._add_generator(circ, self.gen_params)
        self._add_discriminator(circ, self.disc_params)
        return circ

    def get_bloch_vector_real(self, phi, theta, omega):
        """Helper to get Bloch vector for real data (qubit 0)."""
        circ = QuantumCircuit(self.n_qubits)
        self._add_real_data(circ, phi, theta, omega)
        return self._get_bloch_vector(circ)

    def get_bloch_vector_generator(self, gen_w):
        """Helper to get Bloch vector for generator (qubit 0)."""
        circ = QuantumCircuit(self.n_qubits)
        self._add_generator(circ, gen_w)
        return self._get_bloch_vector(circ)

    def _get_bloch_vector(self, circ):
        from qiskit.quantum_info import Statevector
        sv = Statevector.from_instruction(circ)
        X0 = pauli_string_on_qubit("X", 0, self.n_qubits)
        Y0 = pauli_string_on_qubit("Y", 0, self.n_qubits)
        Z0 = pauli_string_on_qubit("Z", 0, self.n_qubits)
        return np.array([
            np.real(sv.expectation_value(X0)),
            np.real(sv.expectation_value(Y0)),
            np.real(sv.expectation_value(Z0))
        ], dtype=float)