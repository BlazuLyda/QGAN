import torch
import numpy as np
import math
from utils import expval_from_statevector
from qiskit import QuantumCircuit
import os
from concurrent.futures import ThreadPoolExecutor

SHIFT = math.pi / 2.0

def tensor_to_bind_dict(tensor, param_vec):
    values = tensor.detach().cpu().double().numpy()
    return {param_vec[i]: values[i] for i in range(len(values))}

def compute_parameter_shift_grads(circuit, measure_op, base_bind, target_params, max_workers=None):
    n = len(target_params)
    grads = np.zeros(n)

    if n == 0:
        return grads

    # default: env var or cpu count
    if max_workers is None:
        max_workers = int(os.getenv("QGAN_WORKERS", "0")) or (os.cpu_count() or 1)

    def one_param(i):
        param = target_params[i]

        bind_plus = base_bind.copy()
        bind_plus[param] += SHIFT
        f_plus = expval_from_statevector(circuit, bind_plus, measure_op)

        bind_minus = base_bind.copy()
        bind_minus[param] -= SHIFT
        f_minus = expval_from_statevector(circuit, bind_minus, measure_op)

        return i, 0.5 * (f_plus - f_minus)

    # For tiny n, threading overhead can dominate
    if n < 4 or max_workers == 1:
        for i in range(n):
            _, g = one_param(i)
            grads[i] = g
        return grads

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, g in ex.map(one_param, range(n)):
            grads[i] = g

    return grads


class RealDiscExpval(torch.autograd.Function):
    @staticmethod
    def forward(ctx, disc_w, qgan, label_val, real_data_fn):
        ctx.qgan = qgan
        full_circ = QuantumCircuit(qgan.disc_qubits)

        # 1. Encode Label on Q1
        if label_val == 1: full_circ.x(1)
        # 2. Encode Real Data on Q2...
        real_qc = real_data_fn(label_val, qgan.n_data)
        if real_qc is not None:
            qubit_map = list(range(2, 2 + qgan.n_data))
            full_circ.compose(real_qc, qubits=qubit_map, inplace=True)
        # 3. Apply Discriminator
        full_circ.compose(qgan.disc_circuit, inplace=True)

        ctx.circuit = full_circ
        bind = tensor_to_bind_dict(disc_w, qgan.disc_params)
        f0 = expval_from_statevector(full_circ, bind, qgan.measure_op)

        ctx.save_for_backward(disc_w)
        return disc_w.new_tensor(f0)

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output = grad_outputs[0]
        (disc_w,) = ctx.saved_tensors
        bind = tensor_to_bind_dict(disc_w, ctx.qgan.disc_params)
        grads = compute_parameter_shift_grads(ctx.circuit, ctx.qgan.measure_op, bind, ctx.qgan.disc_params)
        grad_disc = torch.from_numpy(grads).to(disc_w.device).type_as(disc_w)
        return grad_disc * grad_output, None, None, None

class GenDiscExpval(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gen_w, disc_w_const, qgan, label_val):
        ctx.qgan = qgan
        full_circ = QuantumCircuit(qgan.disc_qubits)

        # 1. Label Q1
        if label_val == 1: full_circ.x(1)
        # 2. Generator (on Q1, Q2)
        gen_map = list(range(1, 1 + qgan.gen_qubits))
        full_circ.compose(qgan.gen_circuit, qubits=gen_map, inplace=True)
        # 3. Discriminator
        full_circ.compose(qgan.disc_circuit, inplace=True)

        ctx.circuit = full_circ
        bind = tensor_to_bind_dict(gen_w, qgan.gen_params)
        bind.update(tensor_to_bind_dict(disc_w_const, qgan.disc_params))
        f0 = expval_from_statevector(full_circ, bind, qgan.measure_op)

        ctx.save_for_backward(gen_w, disc_w_const)
        return gen_w.new_tensor(f0)

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output = grad_outputs[0]
        gen_w, disc_w_const = ctx.saved_tensors
        bind = tensor_to_bind_dict(gen_w, ctx.qgan.gen_params)
        bind.update(tensor_to_bind_dict(disc_w_const, ctx.qgan.disc_params))
        grads = compute_parameter_shift_grads(ctx.circuit, ctx.qgan.measure_op, bind, ctx.qgan.gen_params)
        grad_gen = torch.from_numpy(grads).to(gen_w.device).type_as(gen_w)
        return grad_gen * grad_output, None, None, None