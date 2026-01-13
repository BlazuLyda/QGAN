import numpy as np
import torch
from qiskit import QuantumCircuit
from circuits import QGANCircuits
from gradients import RealDiscExpval, GenDiscExpval
from qiskit.quantum_info import Statevector

# --- Configuration matching Paper Numerics ---
N_DATA_QUBITS = 2
N_LAYERS_GEN = 2    # Paper: 2 layers for Generator
N_LAYERS_DISC = 4   # Paper: 4 layers for Discriminator (Critical for flow Q2->Q1->Q0)
STEPS = 100         # Total Generator steps
DISC_STEPS = 10     # Train Discriminator 10 times per Generator step
LR_G = 0.05
LR_D = 0.05
# ---------------------

def main():
    torch.set_default_dtype(torch.float64)
    np.random.seed(42)
    torch.manual_seed(42)

    # 1. Initialize Conditional QuGAN
    qgan = QGANCircuits(n_data_qubits=N_DATA_QUBITS,
                        n_layers_gen=N_LAYERS_GEN,
                        n_layers_disc=N_LAYERS_DISC)

    print(f"QuGAN Initialized.")
    print(f"Structure: [Dec: 1, Label: 1, Data: 1]")
    print(f"Layers: Gen={N_LAYERS_GEN}, Disc={N_LAYERS_DISC}")
    print(f"Params: Gen={qgan.n_gen_params}, Disc={qgan.n_disc_params}")

    # 2. Define Real Data Source
    # Label 0 -> |00> (Data 0)
    # Label 1 -> |01> (Data 1)
    def get_real_data_circuit(label, n_qubits):
        qc = QuantumCircuit(n_qubits)
        if label == 1:
            for q in range(n_qubits):
                qc.x(q)
        return qc

    # 3. Parameters
    # Generator: Initialize closer to 0 to preserve Label early on.
    gen_w = torch.nn.Parameter(
        torch.tensor(np.random.uniform(-0.1, 0.1, qgan.n_gen_params))
    )
    # Discriminator: Initialize widely [-pi, pi] to avoid barren plateaus.
    disc_w = torch.nn.Parameter(
        torch.tensor(np.random.uniform(-np.pi, np.pi, qgan.n_disc_params))
    )

    opt_g = torch.optim.Adam([gen_w], lr=LR_G)
    opt_d = torch.optim.Adam([disc_w], lr=LR_D)

    print(f"\n--- Starting Training (Target: 0->|00>, 1->|11>) ---")

    for step in range(STEPS):

        # --- A. Train Discriminator (k times) ---
        avg_loss_d = 0
        for _ in range(DISC_STEPS):
            opt_d.zero_grad()
            total_loss_d = 0

            # Sum loss over all labels
            for label in [0, 1]:
                # D(Real) -> Want +1
                exp_real = RealDiscExpval.apply(disc_w, qgan, label, get_real_data_circuit)
                # D(Fake) -> Want -1 (Discriminator distinguishes)
                exp_fake = GenDiscExpval.apply(gen_w.detach(), disc_w, qgan, label)

                # Minimax Loss D: -( E[Real] - E[Fake] )
                # Ideally converges to -2 (if D is perfect and G is bad) or 0 (if G is perfect)
                loss_d_label = -(exp_real - exp_fake)
                total_loss_d += loss_d_label

            total_loss_d.backward()
            opt_d.step()
            avg_loss_d += total_loss_d.item()

        avg_loss_d /= DISC_STEPS

        # --- B. Train Generator (1 time) ---
        opt_g.zero_grad()
        total_loss_g = 0
        for label in [0, 1]:
            # G wants D(Fake) -> +1 (Real)
            exp_fake = GenDiscExpval.apply(gen_w, disc_w.detach(), qgan, label)

            # Minimax Loss G: - E[Fake]
            loss_g_label = -exp_fake
            total_loss_g += loss_g_label

        total_loss_g.backward()
        opt_g.step()

        # --- Logging ---
        if step % 20 == 0:
            print(f"Step {step:03d} | Loss D: {avg_loss_d:.4f} | Loss G: {total_loss_g.item():.4f}")

    # 5. Final Verification
    print("\n--- Final Verification ---")
    print("Explanation: Bitstring is 'Data1 Data0 Label'")
    print("Target Label 0: '00 0'")
    print("Target Label 1: '11 1'")

    for label in [0, 1]:
        qc = QuantumCircuit(qgan.gen_qubits)
        if label == 1: qc.x(0)

        bind = {qgan.gen_params[i]: gen_w.detach().cpu().numpy()[i] for i in range(len(gen_w))}
        qc.compose(qgan.gen_circuit, inplace=True)
        qc = qc.assign_parameters(bind)

        # Get counts/probs
        probs = Statevector(qc).probabilities_dict()

        # Filter small probabilities for clean output
        clean_probs = {k: v for k, v in probs.items() if v > 0.01}
        print(f"Label {label} Output: {clean_probs}")

if __name__ == "__main__":
    main()