import numpy as np
import torch
from utils import prob_from_expval
from circuits import QGANCircuits
from gradients import RealDiscExpval, GenDiscExpval

# --- Configuration ---
N_QUBITS = 3
N_LAYERS = 3
DISC_STEPS = 3
STEPS = 400
LR = 0.01
# ---------------------

def main():
    torch.set_default_dtype(torch.float64)

    # 1. Initialize Circuit Logic
    qgan = QGANCircuits(n_qubits=N_QUBITS, n_layers=N_LAYERS)
    print(f"initialized QGAN with {N_QUBITS} qubits. Parameters per circuit: {qgan.num_params}")

    # 2. Random fixed "Truth" (Real Data)
    np.random.seed(42)
    real_data_values = np.random.uniform(low=-np.pi, high=np.pi, size=(qgan.num_params,))
    real_data_tensor = torch.tensor(real_data_values, requires_grad=False)

    # 3. Helpers
    def prob_real_true(disc_w):
        z = RealDiscExpval.apply(disc_w, real_data_tensor, qgan)
        return prob_from_expval(z)

    def prob_fake_true(gen_w, disc_w):
        z = GenDiscExpval.apply(gen_w, disc_w, qgan)
        return prob_from_expval(z)

    # 4. Costs (Log loss / BCE)
    def disc_cost(gen_w_const, disc_w):
        pr_real = prob_real_true(disc_w)
        pr_fake = prob_fake_true(gen_w_const, disc_w)

        # LSGAN: Pull Real to 1.0, Push Fake to 0.0
        # Cost = (D(real) - 1)^2 + (D(fake) - 0)^2
        return (pr_real - 1.0)**2 + (pr_fake - 0.0)**2

    def gen_cost(gen_w, disc_w_const):
        pr_fake = prob_fake_true(gen_w, disc_w_const)

        # Generator: Pull Fake to 1.0
        # Cost = (D(fake) - 1)^2
        return (pr_fake - 1.0)**2

    # 5. Initialization
    init_gen = np.random.uniform(low=-np.pi, high=np.pi, size=(qgan.num_params,))
    init_disc = np.random.uniform(low=-0.1, high=0.1, size=(qgan.num_params,))

    gen_w = torch.nn.Parameter(torch.tensor(init_gen))
    disc_w = torch.nn.Parameter(torch.tensor(init_disc))

    # --- DIFFERENT LEARNING RATES ---
    opt_d = torch.optim.Adam([disc_w], lr=LR, betas=(0.5, 0.9))
    opt_g = torch.optim.Adam([gen_w], lr=LR, betas=(0.5, 0.9))

    # 6. "k-step" Interleaved Training Loop
    print(f"\n--- Starting Training (Adam + LSGAN) ---")

    for step in range(STEPS):

        # -------------------------
        # A. Update Discriminator (k times)
        # -------------------------
        avg_d_loss = 0
        for _ in range(DISC_STEPS):
            opt_d.zero_grad()
            # We strictly detach Generator here
            loss_d = disc_cost(gen_w.detach(), disc_w)
            loss_d.backward()
            opt_d.step()
            avg_d_loss += loss_d.item()

        avg_d_loss /= DISC_STEPS

        # -------------------------
        # B. Update Generator (1 time)
        # -------------------------
        opt_g.zero_grad()
        # We strictly detach Discriminator here
        loss_g = gen_cost(gen_w, disc_w.detach())
        loss_g.backward()
        opt_g.step()

        # -------------------------
        # C. Logging
        # -------------------------
        if step % 20 == 0:
            with torch.no_grad():
                prob_r = prob_real_true(disc_w).item()
                prob_f = prob_fake_true(gen_w, disc_w).item()
                print(f"Step {step:03d} | Cost D: {avg_d_loss:.4f} | Cost G: {loss_g.item():.4f}")
                print(f"           | P(Real): {prob_r:.4f} | P(Fake): {prob_f:.4f}")

    # 7. Final Analysis
    print("\n--- Final Results ---")
    bv_r = qgan.get_bloch_vector_real(real_data_values)
    bv_g = qgan.get_bloch_vector_generator(gen_w.detach().cpu().numpy())
    dist = np.linalg.norm(bv_r - bv_g)

    print(f"Bloch Vector Distance (Qubit 0): {dist:.4f}")
    print(f"Real Q0: {np.round(bv_r, 3)}")
    print(f"Fake Q0: {np.round(bv_g, 3)}")


if __name__ == "__main__":
    main()