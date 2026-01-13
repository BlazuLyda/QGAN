from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from utils import avg_z_op

class QGANCircuits:
    def __init__(self, n_data_qubits=1, n_layers_gen=2, n_layers_disc=4):
        """
        Implementation of QuGAN with configurable depths.
        Paper (Sec II.E) uses Gen=2 layers, Disc=4 layers.
        """
        self.n_data = n_data_qubits
        self.n_label = 1
        self.n_dec = 1

        # Generator: Acts on Label(1) + Data(n)
        self.gen_qubits = self.n_label + self.n_data
        self.n_layers_gen = n_layers_gen

        # Discriminator: Acts on Decision(1) + Label(1) + Data(n)
        self.disc_qubits = self.n_dec + self.n_label + self.n_data
        self.n_layers_disc = n_layers_disc

        # --- Parameters ---
        self.n_gen_params = self._count_ansatz_params(self.gen_qubits, n_layers_gen)
        self.gen_params = ParameterVector("g", self.n_gen_params)

        self.n_disc_params = self._count_ansatz_params(self.disc_qubits, n_layers_disc)
        self.disc_params = ParameterVector("d", self.n_disc_params)

        # --- Circuits ---
        # Measure Z on Decision qubit (Q0)
        self.measure_op = avg_z_op(total_qubits=self.disc_qubits, target_qubit=0)

        self.gen_circuit = self._build_gen_ansatz()
        self.disc_circuit = self._build_disc_ansatz()

    def _count_ansatz_params(self, n_q, layers):
        # Layer: RX, RZ (2n) + RZZ nearest-neighbor (n-1)
        if n_q > 1:
            params_per_layer = (2 * n_q) + (n_q - 1)
        else:
            params_per_layer = 2 * n_q
        return params_per_layer * layers

    def _add_paper_layer(self, circ, params, qubits):
        idx = 0
        # 1. Single Qubit Rotations (RX, RZ)
        for q in qubits:
            circ.rx(params[idx], q)
            circ.rz(params[idx+1], q)
            idx += 2

        # 2. Entangling Rotations (RZZ)
        if len(qubits) > 1:
            for i in range(len(qubits) - 1):
                circ.rzz(params[idx], qubits[i], qubits[i+1])
                idx += 1
        return idx

    def _build_gen_ansatz(self):
        circ = QuantumCircuit(self.gen_qubits)
        param_idx = 0
        for _ in range(self.n_layers_gen):
            p_count = self._count_ansatz_params(self.gen_qubits, 1)
            layer_params = self.gen_params[param_idx : param_idx + p_count]
            self._add_paper_layer(circ, layer_params, list(range(self.gen_qubits)))
            param_idx += p_count
        return circ

    def _build_disc_ansatz(self):
        circ = QuantumCircuit(self.disc_qubits)
        param_idx = 0
        for _ in range(self.n_layers_disc):
            p_count = self._count_ansatz_params(self.disc_qubits, 1)
            layer_params = self.disc_params[param_idx : param_idx + p_count]
            self._add_paper_layer(circ, layer_params, list(range(self.disc_qubits)))
            param_idx += p_count
        return circ