# gradients.py
import torch
import numpy as np
import math
from utils import expval_from_statevector

SHIFT = math.pi / 2.0

def tensor_to_bind_dict(tensor, param_vec):
    """
    Helper to detach a tensor, convert to numpy, and zip with a ParameterVector.
    """
    values = tensor.detach().cpu().double().numpy()
    return {param_vec[i]: values[i] for i in range(len(values))}

def array_to_bind_dict(numpy_array, param_vec):
    """
    Helper to create a binding dictionary from a numpy array and ParameterVector.
    """
    return {param_vec[i]: numpy_array[i] for i in range(len(numpy_array))}

def compute_parameter_shift_grads(circuit, measure_op, base_bind, target_params):
    """
    Helper to compute gradients for a list/vector of parameters using the parameter-shift rule.
    """
    grads = np.zeros(len(target_params))

    for i, param in enumerate(target_params):
        # Shift +
        bind_plus = base_bind.copy()
        bind_plus[param] += SHIFT
        f_plus = expval_from_statevector(circuit, bind_plus, measure_op)

        # Shift -
        bind_minus = base_bind.copy()
        bind_minus[param] -= SHIFT
        f_minus = expval_from_statevector(circuit, bind_minus, measure_op)

        grads[i] = 0.5 * (f_plus - f_minus)

    return grads

# -----------------------------
# Autograd Functions
# -----------------------------

class RealDiscExpval(torch.autograd.Function):
    """
    Calculates expectation value for the Real Data + Discriminator circuit.
    """
    @staticmethod
    def forward(ctx, disc_w, real_data_w, qgan_circuits):
        """
        Computes forward pass for Real + Discriminator circuit.
        :param ctx: Context for autograd
        :param disc_w: Tensor of trainable Discriminator parameters
        :param real_data_w: Tensor or float array of fixed parameters for real data
        :param qgan_circuits: QGANCircuits object with pre-built circuits and parameters
        :return: Expectation value as a tensor
        """
        ctx.qgan = qgan_circuits

        # Save fixed real params (as numpy) for backward
        real_np = real_data_w.detach().cpu().numpy()
        ctx.real_np = real_np

        # Bind Discriminator (Active)
        bind = tensor_to_bind_dict(disc_w, qgan_circuits.disc_params)

        # Bind Real Data (Fixed)
        bind.update(array_to_bind_dict(real_np, qgan_circuits.real_params))

        f0 = expval_from_statevector(qgan_circuits.real_disc_circuit, bind, qgan_circuits.measure_op)

        ctx.save_for_backward(disc_w)
        return disc_w.new_tensor(f0)

    @staticmethod
    def backward(ctx, *grad_outputs):
        """
        Computes gradients via parameter-shift rule for Discriminator parameters.
        :param ctx: Context with saved tensors and qgan circuits
        :param grad_outputs: Gradient outputs from subsequent layers
        :return: Gradients for disc_w, None for real_data_w and qgan_circuits
        """
        grad_output = grad_outputs[0]
        (disc_w,) = ctx.saved_tensors
        qgan = ctx.qgan
        real_np = ctx.real_np

        base_bind = tensor_to_bind_dict(disc_w, qgan.disc_params)
        base_bind.update(array_to_bind_dict(real_np, qgan.real_params))

        grads = compute_parameter_shift_grads(
            circuit=qgan.real_disc_circuit,
            measure_op=qgan.measure_op,
            base_bind=base_bind,
            target_params=qgan.disc_params
        )

        grad_disc = torch.from_numpy(grads).to(disc_w.device).type_as(disc_w)

        # Return None for real_data_w and qgan_circuits
        return grad_disc * grad_output, None, None


class GenDiscExpval(torch.autograd.Function):
    """
    Calculates expectation value for the Generator + Discriminator circuit.
    """
    @staticmethod
    def forward(ctx, gen_w, disc_w_const, qgan_circuits):
        ctx.qgan = qgan_circuits

        # Combine bindings for Generator and Discriminator
        bind = tensor_to_bind_dict(gen_w, qgan_circuits.gen_params)
        bind.update(tensor_to_bind_dict(disc_w_const, qgan_circuits.disc_params))

        f0 = expval_from_statevector(qgan_circuits.gen_disc_circuit, bind, qgan_circuits.measure_op)

        ctx.save_for_backward(gen_w, disc_w_const)
        return gen_w.new_tensor(f0)

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output = grad_outputs[0]
        gen_w, disc_w_const = ctx.saved_tensors
        qgan = ctx.qgan

        # Reconstruct the binding dictionary
        base_bind = tensor_to_bind_dict(gen_w, qgan.gen_params)
        base_bind.update(tensor_to_bind_dict(disc_w_const, qgan.disc_params))

        # Compute gradients via parameter shift
        grads = compute_parameter_shift_grads(
            circuit=qgan.gen_disc_circuit,
            measure_op=qgan.measure_op,
            base_bind=base_bind,
            target_params=qgan.gen_params
        )

        grad_gen = torch.from_numpy(grads).to(gen_w.device).type_as(gen_w)

        return grad_gen * grad_output, None, None
