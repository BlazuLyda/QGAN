from data import QuantumDataSource, QuantumEnsemble, create_cx_data_source
from gradients import tensor_to_bind_dict
from qiskit.quantum_info import Statevector

from train import TrainQGAN, TrainingConfig

# --- Configuration matching Paper Numerics ---
N_LAYERS_GEN = 2    # Paper: 2 layers for Generator
N_LAYERS_DISC = 4   # Paper: 4 layers for Discriminator (Critical for flow Q2->Q1->Q0)
STEPS = 100         # Total Generator steps
DISC_STEPS = 10     # Train Discriminator 10 times per Generator step
GEN_STEPS = 2       # Train Generator 2 times per iteration
LR_G = 0.25
LR_D = 0.05
# ---------------------

def init_cx_training():

    config = TrainingConfig(
        n_qubits_bath=0,
        n_layers_gen=N_LAYERS_GEN,
        n_layers_disc=N_LAYERS_DISC,
        iterations=STEPS,
        disc_steps=DISC_STEPS,
        gen_steps=1,
        lr_g=LR_G,
        lr_d=LR_D,
        batch_size=1, # Used data source has pure classes, thus no bath register and no randomness
        seed=42
    )

    training = TrainQGAN(
        real_data=create_cx_data_source(),
        config=config
    )

    return training


def init_simple_training():

    config = TrainingConfig(
        n_qubits_bath=0,
        n_layers_gen=1,
        n_layers_disc=3,
        iterations=STEPS,
        disc_steps=DISC_STEPS,
        gen_steps=1,
        lr_g=LR_G,
        lr_d=LR_D,
        batch_size=1, # Used data source has pure classes, thus no bath register and no randomness
        seed=42
    )

    # Single class data source |+> = ( |0> + |1> ) / sqrt(2)
    ens_0 = QuantumEnsemble(
        states=[Statevector.from_label('+')],
        probs=[1.0]
    )
    data = QuantumDataSource(
        n_qubits_data=1,
        n_qubits_label=1,
        ensembles=[ens_0]
    )

    training = TrainQGAN(
        real_data=data,
        config=config
    )

    return training

def main():

    # Piece of code that might be useful later
    # --- Logging ---
    # if step % 20 == 0:
    #     print(f"Step {step:03d} | Loss D: {avg_loss_d:.4f} | Loss G: {total_loss_g.item():.4f}")

    # config, training = init_cx_training()
    training = init_simple_training()
    print("Training initialized. Starting training...")

    result = training.run()

    # Final Verification
    print("--- Training Finished ---")

    print(f"Trained Generator Parameters:\n{result.gen_params}")
    print(f"Trained Discriminator Parameters:\n{result.disc_params}")

    print("\n--- Generated Samples from Trained Generator ---")

    # For each class, sample states from the trained Generator
    n_samples = 3
    gen_param_bind = tensor_to_bind_dict(
        result.gen_params,
        training.gen_sampler.gen_params
    )

    for label in range(training.real_data.num_classes):
        print(f"\nExpected density matrix for class {label}:")
        print(training.real_data.get_ensemble(label).get_density_matrix())

        print(f"\nGenerated samples for class {label}:")
        for _ in range(n_samples):
            gen_dm = training.gen_sampler.sample_data_density_matrix(label, gen_param_bind)
            print(gen_dm)
        

if __name__ == "__main__":
    main()


