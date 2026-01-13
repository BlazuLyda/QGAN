# main.py
import math
import numpy as np
import torch
from utils import prob_from_expval
from circuits import QGANCircuits
from gradients import RealDiscExpval, GenDiscExpval

def main():
    torch.set_default_dtype(torch.float64)

    # 1. Initialize Circuit Logic
    qgan = QGANCircuits()

    # 2. Configuration (Data Angles)
    phi = math.pi / 6.0
    theta = math.pi / 2.0
    omega = math.pi / 7.0

    # 3. Helpers for Probabilities (Closure style to inject 'qgan')
    def prob_real_true(disc_w):
        # Pass 'qgan' into the custom autograd function
        z = RealDiscExpval.apply(disc_w, phi, theta, omega, qgan)
        return prob_from_expval(z)

    def prob_fake_true(gen_w, disc_w):
        # Pass 'qgan' into the custom autograd function
        z = GenDiscExpval.apply(gen_w, disc_w, qgan)
        return prob_from_expval(z)

    # 4. Cost Functions
    def disc_cost(gen_w_const, disc_w):
        # Cost_D = Pr(real|fake) - Pr(real|real)
        return prob_fake_true(gen_w_const, disc_w) - prob_real_true(disc_w)

    def gen_cost(gen_w, disc_w_const):
        # Cost_G = -Pr(real|fake)
        return -prob_fake_true(gen_w, disc_w_const)

    # 5. Initialization
    np.random.seed(0)
    eps = 1e-2
    init_gen = np.array([math.pi] + [0.0] * 8) + np.random.normal(scale=eps, size=(9,))
    init_disc = np.random.normal(size=(9,))

    gen_w = torch.nn.Parameter(torch.tensor(init_gen))
    disc_w = torch.nn.Parameter(torch.tensor(init_disc))

    # -------------------------
    # Stage 1: Train Discriminator
    # -------------------------
    opt_d = torch.optim.SGD([disc_w], lr=0.4)
    print("--- Training Discriminator ---")
    for step in range(50):
        opt_d.zero_grad()
        loss = disc_cost(gen_w.detach(), disc_w)
        loss.backward()
        opt_d.step()
        if step % 5 == 0:
            print(f"[D] Step {step:02d}: cost = {loss.item(): .6f}")

    pr_rt = prob_real_true(disc_w).item()
    pr_ft = prob_fake_true(gen_w.detach(), disc_w).item()
    print("\nAfter D training:")
    print("Prob(real classified as real): ", pr_rt)
    print("Prob(fake classified as real): ", pr_ft)

    # -------------------------
    # Stage 2: Train Generator
    # -------------------------
    opt_g = torch.optim.SGD([gen_w], lr=0.4)
    disc_fixed = disc_w.detach()

    print("\n--- Training Generator ---")
    for step in range(50):
        opt_g.zero_grad()
        loss = gen_cost(gen_w, disc_fixed)
        loss.backward()
        opt_g.step()
        if step % 5 == 0:
            print(f"[G] Step {step:02d}: cost = {loss.item(): .6f}")

    # 6. Final Analysis
    pr_ft2 = prob_fake_true(gen_w.detach(), disc_fixed).item()
    disc_cost_final = (prob_fake_true(gen_w.detach(), disc_fixed) -
                       prob_real_true(disc_fixed)).item()

    print("\nAfter G training:")
    print("Prob(fake classified as real): ", pr_ft2)
    print("Discriminator cost: ", disc_cost_final)

    bv_r = qgan.get_bloch_vector_real(phi, theta, omega)
    bv_g = qgan.get_bloch_vector_generator(gen_w.detach().cpu().numpy())

    print("\nBloch vectors on qubit 0:")
    print("Real Bloch vector:      ", bv_r)
    print("Generator Bloch vector: ", bv_g)

if __name__ == "__main__":
    main()