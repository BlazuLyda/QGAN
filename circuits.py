from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from utils import avg_z_op

class Ansatz:
    @staticmethod
    def count_ansatz_params(n_qubits, n_layers):
        # Layer: RX, RZ (2n) + RZZ nearest-neighbor (n-1)
        if n_qubits > 1:
            params_per_layer = (2 * n_qubits) + (n_qubits - 1)
        else:
            params_per_layer = 2 * n_qubits
        return params_per_layer * n_layers

    @staticmethod
    def add_ansatz(qubits, n_layers: int, circ: QuantumCircuit, params):
        """
        Add the scalable ansatz unitary with a specified number of layers
        to the selected qubits of a quantum circuit. 
        """

        if Ansatz.count_ansatz_params(len(qubits), n_layers) != len(params):
            raise Exception("Number of supplied parameter values doesn't match required number of params")

        param_idx = 0
        for _ in range(n_layers):
            # 1. Single Qubit Rotations (RX, RZ)
            for q in qubits:
                circ.rx(params[param_idx], q)
                param_idx += 1
                circ.rz(params[param_idx], q)
                param_idx += 1

            # 2. Entangling Rotations (RZZ)
            if len(qubits) > 1:
                for i in range(len(qubits) - 1):
                    circ.rzz(params[param_idx], qubits[i], qubits[i+1])
                    param_idx += 1

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
        self.n_gen_params = Ansatz.count_ansatz_params(self.gen_qubits, n_layers_gen)
        self.gen_params = ParameterVector("g", self.n_gen_params)

        self.n_disc_params = Ansatz.count_ansatz_params(self.disc_qubits, n_layers_disc)
        self.disc_params = ParameterVector("d", self.n_disc_params)

        # --- Circuits ---
        # Measure Z on Decision qubit (Q0)
        self.measure_op = avg_z_op(total_qubits=self.disc_qubits, target_qubit=0)

        self.gen_circuit = self._build_gen_ansatz()
        self.disc_circuit = self._build_disc_ansatz()

    def _build_gen_ansatz(self):
        circ = QuantumCircuit(self.gen_qubits)
        Ansatz.add_ansatz(list(range(self.gen_qubits)), self.n_layers_gen, circ, self.gen_params)
        return circ

    def _build_disc_ansatz(self):
        circ = QuantumCircuit(self.disc_qubits)
        Ansatz.add_ansatz(list(range(self.disc_qubits)), self.n_layers_disc, circ, self.disc_params)
        return circ


class GenCircuits:
    """
    Quantum circuit that uses the Generator unitary with supplied static parameters
    to generate artificial samples of learned data.
    """

    def __init__(self, gen_params, n_qubits=3, n_layers=2):
        """
        :param gen_w: Learned parameters for generator
        :param n_qubits: Number of qubits for generator
        :param n_layers: Depth of the generator ansatz
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # We'll use an ansatz with 3 params per qubit per layer (RX, RY, RZ)
        self.params_per_qubit = 3

        # Calculate total parameters needed
        self.num_params = self.n_qubits * self.n_layers * self.params_per_qubit

        if self.num_params != len(gen_params):
            raise Exception("Number of supplied parameter values doesn't match required number of params")

        # Define Parameter Vectors
        self.gen_params = gen_params
        self.gen_circuit = self._build_gen()

    def _build_gen(self):
        circ = QuantumCircuit(self.n_qubits)
        # add_scalable_ansatz(self.n_layers, self.n_qubits, circ, self.gen_params)    # Generator
        return circ

    def sample_statevector(self, label: int, rand: int|None = None):
        """
        Evaluate the generator quantum circuit and return the generated result.
        
        :param self: Description
        :param label: Description
        :type label: int
        :param rand: Description
        :type rand: int | None
        """
