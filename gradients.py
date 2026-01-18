import torch
import numpy as np
import math
from typing import Dict, Any, Tuple, Callable
from qiskit.circuit import ParameterVector

from circuits import QGANCircuits

SHIFT: float = math.pi / 2.0


def tensor_to_bind_dict(
    tensor: torch.Tensor, param_vec: ParameterVector
) -> Dict[Any, float]:
    """
    Maps a PyTorch tensor of values to a Qiskit Parameter dictionary.

    Args:
        tensor: The weights/parameters from the torch model.
        param_vec: The Qiskit ParameterVector associated with these weights.

    Returns:
        A dictionary suitable for circuit.assign_parameters().
    """
    values = tensor.detach().cpu().double().numpy()
    return {param_vec[i]: float(values[i]) for i in range(len(values))}


def compute_parameter_shift_grads(
    calc_circ_expval: Callable[[Any], float],
    base_bind: Dict[Any, float],
    target_params: ParameterVector,
) -> np.ndarray:
    """
    Computes gradients for a set of parameters using the Parameter Shift Rule.

    Args:
        calc_circ_expval: Function that takes a parameter binding dictionary
            and returns the expectation value (float).
        base_bind: Current parameter bindings (theta).
        target_params: The subset of parameters to compute gradient for.

    Returns:
        A numpy array containing first-order derivatives (gradient).
    """
    n = len(target_params)
    grads = np.zeros(n)

    for i in range(n):
        param = target_params[i]

        # f(theta + pi/2)
        bind_plus = base_bind.copy()
        bind_plus[param] += SHIFT
        f_plus = calc_circ_expval(bind_plus)

        # f(theta - pi/2)
        bind_minus = base_bind.copy()
        bind_minus[param] -= SHIFT
        f_minus = calc_circ_expval(bind_minus)

        grads[i] = 0.5 * (f_plus - f_minus)

    return grads


class Gradients:
    """
    Container for gradient computation methods for QGAN training.
    """

    @staticmethod
    def RD_disc_grad(
        disc_w: torch.Tensor, 
        qgan: QGANCircuits, 
        label_val: int
    ) -> Tuple[torch.Tensor, float]:
        """
        Computes the gradient of the Z expectation of the Real-Discriminator
        circuit with respect to the Discriminator parameters.

        Args:
            disc_w: Discriminator weights as a torch Tensor.
            qgan: The QGANCircuits instance.
            label_val: The class label value
        """

        # Convert Pytorch Tensor to Qiskit parameter binding dictionary
        base_bind = tensor_to_bind_dict(disc_w, qgan.disc_params)

        # Initialize the Real-Discriminator circuit evaluator
        calc_RD_expval = qgan.prepare_RD_circuit(label_val)
        assert isinstance(calc_RD_expval, Callable)

        # Compute base bind expectation value
        expval = calc_RD_expval(base_bind)

        # Compute Discriminator gradients using Parameter Shift Rule
        grads = compute_parameter_shift_grads(
            calc_RD_expval, base_bind, qgan.disc_params
        )

        # Convert to torch Tensor
        grad_disc = torch.from_numpy(grads).to(disc_w.device).type_as(disc_w)

        return grad_disc, expval


    @staticmethod
    def GD_disc_grad(
        disc_w: torch.Tensor, 
        gen_w_const: torch.Tensor, 
        qgan: QGANCircuits, 
        label_val: int
    ) -> Tuple[torch.Tensor, float]:
        """
        Computes the gradient of the Z expectation of the Generator-Discriminator
        circuit with respect to the Discriminator parameters.

        Args:
            disc_w: Discriminator weights as a torch Tensor.
            gen_w: Generator weights as a torch Tensor.
            qgan: The QGANCircuits instance.
            label_val: The class label value
        """

        # Convert Pytorch Tensor to Qiskit parameter binding dictionary
        base_bind = tensor_to_bind_dict(disc_w, qgan.disc_params)
        base_bind.update(tensor_to_bind_dict(gen_w_const, qgan.gen_params))

        # Initialize the Generator-Discriminator circuit evaluator
        calc_RD_expval = qgan.prepare_GD_circuit(label_val)
        assert isinstance(calc_RD_expval, Callable)

        # Compute base bind expectation value
        expval = calc_RD_expval(base_bind)

        # Compute Discriminator gradients using Parameter Shift Rule
        grads = compute_parameter_shift_grads(
            calc_RD_expval, base_bind, qgan.disc_params
        )

        # Convert to torch Tensor
        grad_disc = torch.from_numpy(grads).to(disc_w.device).type_as(disc_w)

        return grad_disc, expval


    @staticmethod
    def GD_gen_grad(
        gen_w: torch.Tensor,
        disc_w_const: torch.Tensor,
        qgan: QGANCircuits,
        label_val: int
    ) -> Tuple[torch.Tensor, float]:
        """
        Computes the gradient of the Z expectation of the Generator-Discriminator
        circuit with respect to the Generator parameters.

        Args:
            gen_w: Generator weights as a torch Tensor.
            disc_w_const: Discriminator weights as a torch Tensor (constant).
            qgan: The QGANCircuits instance.
            label_val: The class label value
        """

        # Convert Pytorch Tensor to Qiskit parameter binding dictionary
        base_bind = tensor_to_bind_dict(gen_w, qgan.gen_params)
        base_bind.update(tensor_to_bind_dict(disc_w_const, qgan.disc_params))

        # Initialize the Generator-Discriminator circuit evaluator
        calc_GD_expval = qgan.prepare_GD_circuit(label_val)
        assert isinstance(calc_GD_expval, Callable)

        # Compute base bind expectation value
        expval = calc_GD_expval(base_bind)

        # Compute Generator gradients using Parameter Shift Rule
        grads = compute_parameter_shift_grads(
            calc_GD_expval, base_bind, qgan.gen_params
        )

        # Convert to torch Tensor
        grad_gen = torch.from_numpy(grads).to(gen_w.device).type_as(gen_w)

        return grad_gen, expval
