import torch
import numpy as np
import math
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Any, Tuple, Callable
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
    max_workers: Optional[int] = None,
) -> np.ndarray:
    """
    Computes gradients for a set of parameters using the Parameter Shift Rule.

    Args:
        calc_circ_expval: Function that takes a parameter binding dictionary
            and returns the expectation value (float).
        base_bind: Current parameter bindings (theta).
        target_params: The subset of parameters to compute gradient for.
        max_workers: Number of threads for parallel evaluation.

    Returns:
        A numpy array containing first-order derivatives (gradient).
    """
    n = len(target_params)
    grads = np.zeros(n)

    if n == 0:
        return grads

    if max_workers is None:
        max_workers = int(os.getenv("QGAN_WORKERS", "0")) or (os.cpu_count() or 1)

    def one_param(i: int) -> Tuple[int, float]:
        param = target_params[i]

        # f(theta + pi/2)
        bind_plus = base_bind.copy()
        bind_plus[param] += SHIFT
        f_plus = calc_circ_expval(bind_plus)

        # f(theta - pi/2)
        bind_minus = base_bind.copy()
        bind_minus[param] -= SHIFT
        f_minus = calc_circ_expval(bind_minus)

        return i, 0.5 * (f_plus - f_minus)

    if n < 4 or max_workers == 1:
        for i in range(n):
            _, g = one_param(i)
            grads[i] = g
        return grads

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, g in ex.map(one_param, range(n)):
            grads[i] = g

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



class RealDiscExpval(torch.autograd.Function):
    """
    Autograd interface for evaluating the Discriminator on Real data.
    """

    @staticmethod
    def forward(
        ctx: Any, disc_w: torch.Tensor, qgan: QGANCircuits, label_val: int
    ) -> torch.Tensor:
        """Computes the forward pass (expectation value) for Real data."""
        ctx.qgan = qgan

        ctx.calc_RD_expval = qgan.prepare_RD_circuit(label_val)

        bind = tensor_to_bind_dict(disc_w, qgan.disc_params)
        assert isinstance(ctx.calc_RD_expval, Callable)
        f0 = ctx.calc_RD_expval(bind)

        ctx.save_for_backward(disc_w)
        return disc_w.new_tensor(f0)

    @staticmethod
    def backward(
        ctx: Any, *grad_outputs: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[None], Optional[None]]:
        """
        Backward pass for Real-Discriminator circuit.
        Returns 3 values to match forward(disc_w, qgan, label_val).
        """
        grad_output = grad_outputs[0]
        (disc_w,) = ctx.saved_tensors
        bind = tensor_to_bind_dict(disc_w, ctx.qgan.disc_params)

        grads = compute_parameter_shift_grads(
            ctx.calc_RD_expval, bind, ctx.qgan.disc_params
        )

        grad_disc = torch.from_numpy(grads).to(disc_w.device).type_as(disc_w)
        # Returns grad for disc_w, None for qgan, None for label_val
        return grad_disc * grad_output, None, None


class GenDiscExpval(torch.autograd.Function):
    """
    Autograd interface for evaluating the Discriminator on Generated data.
    This function primarily updates the Generator weights.
    """

    @staticmethod
    def forward(
        ctx: Any,
        gen_w: torch.Tensor,
        disc_w_const: torch.Tensor,
        qgan: Any,
        label_val: int,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Computes the forward pass for Generated data."""
        ctx.qgan = qgan

        rng = np.random.default_rng(seed)
        ctx.calc_GD_expval = qgan.prepare_GD_circuit(label_val, rng)

        bind = tensor_to_bind_dict(gen_w, qgan.gen_params)
        bind.update(tensor_to_bind_dict(disc_w_const, qgan.disc_params))

        f0 = ctx.calc_GD_expval(bind)

        ctx.save_for_backward(gen_w, disc_w_const)
        return gen_w.new_tensor(f0)

    @staticmethod
    def backward(
        ctx: Any, *grad_outputs: torch.Tensor
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[None],
        Optional[None],
        Optional[None],
        Optional[None],
    ]:
        """
        Backward pass for Generator-Discriminator circuit.
        Returns 5 values to match forward(gen_w, disc_w_const, qgan, label_val, seed).
        """
        grad_output = grad_outputs[0]
        gen_w, disc_w_const = ctx.saved_tensors
        bind = tensor_to_bind_dict(gen_w, ctx.qgan.gen_params)
        bind.update(tensor_to_bind_dict(disc_w_const, ctx.qgan.disc_params))

        grads = compute_parameter_shift_grads(
            ctx.calc_GD_expval, bind, ctx.qgan.gen_params
        )

        grad_gen = torch.from_numpy(grads).to(gen_w.device).type_as(gen_w)
        # Returns grad for gen_w, and None for all other forward inputs
        return grad_gen * grad_output, None, None, None, None
