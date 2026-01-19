from typing import Any, Callable
from matplotlib.pylab import qr
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace, SparsePauliOp
from data import QuantumDataSource
from utils import avg_z_op, double_qubit_op
import numpy as np

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
    """
    QGAN circuits consisting of a data source G(enerator)/R(eal) and D(iscriminator).
    The layouts of the circuits is presented below.

    Generator - Discriminator Circuit (GD):
    |decision> (1) ------------|^^^^^|------{ measure Z }
    |label   > (m) ------------|  D  |--/
    |data    > (n) ---|^^^^^|--|_____|--/
    |label   > (m) ---|  G  |----/
    |bath    > (k) ---|_____|----/

    Real - Discriminator Circuit (RD):
    |decision> (1) ------------|^^^^^|------{ measure Z }
    |label   > (m) ------------|  D  |--/
    |data    > (n) ---|^^^^^|--|_____|--/
                      |  R  |
                      |_____|
    """

    def __init__(
            self,
            real_source: QuantumDataSource,
            n_data_qubits=1,
            n_label_qubits=1,
            n_bath_qubits=1,
            n_layers_gen=2,
            n_layers_disc=4,
            measure_op:Callable[..., SparsePauliOp]=avg_z_op,
            random:int|None=None
        ):
        """
        Implementation of QuGAN with configurable depths.
        Paper (Sec II.E) uses Gen=2 layers, Disc=4 layers.
        """
        self.real_source = real_source
        self.rng = np.random.default_rng(random)

        self.n_data = n_data_qubits
        self.n_label = n_label_qubits
        self.n_bath = n_bath_qubits
        self.n_dec = 1

        # Offsets of the qubit registers
        self.offsets = {
            "dec":     0,
            "label_d": self.n_dec,
            "data":    self.n_dec + self.n_label,
            "label_g": self.n_dec + self.n_label + self.n_data,
            "bath":    self.n_dec + self.n_label + self.n_data + self.n_label,
        }

        # Generator: Acts on Label(m) + Data(n) + Entropy/Bath(k)
        self.n_gen_qubits = self.n_label + self.n_data + self.n_bath
        self.gen_qubits = list(range(self.offsets["data"], self.offsets["data"] + self.n_gen_qubits))
        self.n_layers_gen = n_layers_gen

        # Discriminator: Acts on Decision(1) + Label(m) + Data(n)
        self.n_disc_qubits = self.n_dec + self.n_label + self.n_data
        self.disc_qubits = list(range(self.n_disc_qubits))
        self.n_layers_disc = n_layers_disc

        # --- Parameters ---
        self.n_gen_params = Ansatz.count_ansatz_params(self.n_gen_qubits, n_layers_gen)
        self.gen_params = ParameterVector("g", self.n_gen_params) # Bindable parameters for Generator

        self.n_disc_params = Ansatz.count_ansatz_params(self.n_disc_qubits, n_layers_disc)
        self.disc_params = ParameterVector("d", self.n_disc_params) # Bindable parameters for Discriminator

        # --- Circuits ---
        # Sizes of the full circuits
        self.n_RD_qubits = self.n_disc_qubits
        self.n_GD_qubits = self.n_disc_qubits + self.n_label + self.n_bath

        # Measure Z on Decision qubit (Q0)
        # self.measure_op_rd = measure_op(self.n_disc_qubits, 0)
        # self.measure_op_gd = measure_op(self.n_disc_qubits + self.n_label + self.n_bath, 0)

        # Measure Pauli on Decision qubit (Q0) and Data qubit
        self.measure_op_rd = double_qubit_op(
            total_qubits=self.n_RD_qubits, 
            decision_qubit=0,
            other_qubit=self.offsets["data"],
            lambda_ZZ=0.02,
            lambda_XX=0.02
        )
        self.measure_op_gd = double_qubit_op(
            total_qubits=self.n_GD_qubits, 
            decision_qubit=0,
            other_qubit=self.offsets["data"],
            lambda_ZZ=0.02,
            lambda_XX=0.02
        )

        self.gen_circuit = self._build_gen_ansatz()
        self.disc_circuit = self._build_disc_ansatz()

    def _build_gen_ansatz(self):
        circ = QuantumCircuit(self.n_gen_qubits)
        Ansatz.add_ansatz(list(range(self.n_gen_qubits)), self.n_layers_gen, circ, self.gen_params)
        return circ

    def _build_disc_ansatz(self):
        circ = QuantumCircuit(self.n_disc_qubits)
        Ansatz.add_ansatz(list(range(self.n_disc_qubits)), self.n_layers_disc, circ, self.disc_params)
        return circ

    def _prepare_label_register(self, circ: QuantumCircuit, label: int, prepare_g: bool = False):
        """
        Convert the label to a binary string and flip all 1's using X gates.
        There are 2 individual bath registers, one for D and one for G.

        :param circ: Quantum circuit to modify
        :type circ: QuantumCircuit
        :param label: Numerical label to prepare
        :type label: int
        :param prepare_g: Whether to prepare the G label register as well
        :type prepare_g: bool
        """
        bits = format(label, f"0{self.n_label}b")[::-1]
        for i, bit in enumerate(bits):
            if bit == "1":
                circ.x(self.offsets["label_d"] + i)
                if prepare_g:
                    circ.x(self.offsets["label_g"] + i)

    def _prepare_bath_register(self, circ: QuantumCircuit):
        """
        Initialize bath qubits to random pure product states.
        """
        offset = self.offsets["bath"]

        for q in range(self.n_bath):
            theta = self.rng.uniform(0, np.pi)
            phi = self.rng.uniform(0, 2 * np.pi)

            circ.ry(theta, offset + q)
            circ.rz(phi, offset + q)

    def _prepare_real_data_register(self, circ: QuantumCircuit, label: int):
        """
        In case of Real data source, prepare the real data register.
        """
        sample_state = self.real_source.sample_class(label, self.rng)
        prep = StatePreparation(sample_state)

        data_offset = self.offsets["data"]
        data_qubits = list(range(data_offset, data_offset + self.n_data))
        circ.append(prep, data_qubits)
        return sample_state

    def prepare_RD_circuit(self, label: int, return_circ: bool = False):
        """
        Evaluate the Real vs Discriminator circuit and return the expectation value.

        :param label: Label of the class to be sampled from
        :type label: int
        """
        circ = QuantumCircuit(self.n_RD_qubits)

        # 1. Decision qubit is already |0>

        # 2. Label register
        self._prepare_label_register(circ, label)

        # 3. Data register (Real data)
        sample_state = self._prepare_real_data_register(circ, label)

        circ.barrier(list(range(self.n_RD_qubits)))

        # 4. Apply Discriminator (symbolic parameters)
        circ.compose(self.disc_circuit, inplace=True)

        if return_circ:
            return circ

        def apply(binds, print_circ: bool = False):
            bound_circ = circ.assign_parameters(binds)
            if print_circ:
                print(f"\n--- RD Circuit (label: {label}) ---\n")
                print(f"Sample state: {sample_state}\n")
                print(bound_circ.draw(fold=-1))

            # 5. Exact simulation
            state = Statevector.from_instruction(bound_circ)
            expval = state.expectation_value(self.measure_op_rd)

            return expval.real

        return apply


    def prepare_GD_circuit(self, label: int, return_circ: bool = False):
        """
        Evaluate the Generator vs Discriminator circuit and return the expectation value.

        :param label: Label of the class to be sampled from
        """
        circ = QuantumCircuit(self.n_GD_qubits)

        # 1. Decision qubit is already |0>

        # 2. Both label registers
        self._prepare_label_register(circ, label, prepare_g=True)

        # 3. Bath register (random noise)
        self._prepare_bath_register(circ)

        circ.barrier(list(range(self.n_GD_qubits)))

        # 4. Apply Generator (symbolic parameters)
        circ.compose(self.gen_circuit, qubits=self.gen_qubits, inplace=True)

        circ.barrier(list(range(self.n_GD_qubits)))

        # 4. Apply Discriminator (symbolic parameters)
        circ.compose(self.disc_circuit, qubits=self.disc_qubits, inplace=True)

        if return_circ:
            return circ

        def apply(binds, print_circ: bool = False):
            bound_circ = circ.assign_parameters(binds)

            if print_circ:
                print(f"\n--- GD Circuit (label: {label}) ---\n")
                print(bound_circ.draw(fold=-1))

            # 5. Exact simulation
            state = Statevector.from_instruction(bound_circ)
            expval = state.expectation_value(self.measure_op_gd)

            return expval.real

        return apply


class GenCircuit:
    """
    Quantum circuit that uses the Generator unitary with supplied static parameters
    to generate artificial samples of learned data.

    The Generator unitary G takes the following quantum registers with variable num of qubits:
    1. Label register for selecting the class to generate the samples from - initialized to the label value.
    2. Data register - initialized to zeros.
    3. Bath register - initialized to random noise.

    The size of the label and data registers is defined by the training data. The bath/entropy register's size can
    be set to:
    - `= n_data_qubits` - minimal expressive latent space
    - `> n_data_qubits` - higher entropy, more expressivity
    - `= 0` - Deterministic generator (usually not desired)

    The noise type used for the Bath/Entropy register is Haar-random product state. Each qubit of the register is
    initialized to the pure state: ∣ψ⟩ = cos(θ/2)|0⟩ + exp(iϕ) sin(θ/2)|1⟩, where parameters θ and ϕ are uniformly
    sampled.
    """

    def __init__(
        self,
        n_data_qubits=1,
        n_label_qubits=1,
        n_bath_qubits=1,
        n_layers=2,
        random:int|None=None
    ):
        """
        :param gen_w: Learned parameters for generator
        :param n_qubits: Number of qubits for generator
        :param n_layers: Depth of the generator ansatz
        """
        self.rng = np.random.default_rng(random)

        self.n_data = n_data_qubits
        self.n_label = n_label_qubits
        self.n_bath = n_bath_qubits

        self.n_qubits = n_data_qubits + n_label_qubits + n_bath_qubits
        self.n_layers = n_layers

        # Parameters for Generator
        self.n_gen_params = Ansatz.count_ansatz_params(self.n_qubits, self.n_layers)
        self.gen_params = ParameterVector("g", self.n_gen_params) # Bindable parameters for Generator

        # Quantum Circuit
        self.gen_circuit = self._build_gen()

    def _build_gen(self):
        circ = QuantumCircuit(self.n_qubits)
        Ansatz.add_ansatz(list(range(self.n_qubits)), self.n_layers, circ, self.gen_params)
        return circ

    def _prepare_label_register(self, circ: QuantumCircuit, label: int):
        """
        Convert the label to a binary string and flip all 1's using X gates.
        """
        bits = format(label, f"0{self.n_label}b")[::-1]
        for i, bit in enumerate(bits):
            if bit == "1":
                circ.x(i)

    def _prepare_bath_register(self, circ: QuantumCircuit):
        """
        Initialize bath qubits to random pure product states.
        """
        offset = self.n_label + self.n_data

        for q in range(self.n_bath):
            theta = self.rng.uniform(0, np.pi)
            phi = self.rng.uniform(0, 2 * np.pi)

            circ.ry(theta, offset + q)
            circ.rz(phi, offset + q)

    def trace_out_bath_and_label(self, state: Statevector) -> DensityMatrix:
        """
        Given the full generated statevector, trace out the bath and label qubits.
        """
        # Indices of qubits to trace out
        trace_out = (
            list(range(self.n_label)) +
            list(range(self.n_label + self.n_data, self.n_qubits))
        )
        # Full and reduced density matrices
        rho_full = DensityMatrix(state)
        rho_data = partial_trace(rho_full, trace_out)

        return rho_data

    def prepare_circuit(self, label: int) -> QuantumCircuit:
        """
        Prepare the full parameterized Generator circuit with label and bath initialization.
        The initialization includes random bath state preparation.
        """
        circ = QuantumCircuit(self.n_qubits)

        # 1. Label register
        self._prepare_label_register(circ, label)

        # 2. Data register is already |0...0⟩

        # 3. Bath register (random noise)
        self._prepare_bath_register(circ)

        # 4. Apply Generator
        circ.compose(self.gen_circuit, inplace=True)

        return circ


    def sample_statevector(self, label: int, binds):
        """
        Evaluate the generator circuit and return the full generated statevector.

        :param label: Label of the class to be sampled from
        :param binds: Parameter bindings for the generator circuit
        """
        # 1. Prepare the parameterized circuit
        circ = self.prepare_circuit(label)

        # 2. Bind parameters
        bound_circ = circ.assign_parameters(binds)

        # 3. Exact simulation: get final statevector
        return Statevector.from_instruction(bound_circ)


    def sample_data_density_matrix(self, label: int, binds):
        """
        Evaluate the generator circuit and return the reduced density matrix
        over the data qubits (after tracing out bath and label qubits).

        :param label: Label of the class to be sampled from
        :param binds: Parameter bindings for the generator circuit
        """

        full_state = self.sample_statevector(label, binds)
        reduced_rho = self.trace_out_bath_and_label(full_state)

        return reduced_rho
