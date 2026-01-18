from data import QuantumDataSource, QuantumEnsemble, create_cx_data_source
from gradients import tensor_to_bind_dict
from qiskit.quantum_info import Statevector

from train import TrainQGAN, TrainingConfig
from evaluation import compute_cross_entropy_over_labels_tensor

# --- Configuration matching Paper Numerics ---
N_LAYERS_GEN = 2    # Paper: 2 layers for Generator
N_LAYERS_DISC = 4   # Paper: 4 layers for Discriminator (Critical for flow Q2->Q1->Q0)
DISC_STEPS = 10     # Train Discriminator 10 times per Generator step
GEN_STEPS = 2       # Train Generator 2 times per iteration
LR_G = 0.15
LR_D = 0.04
# ---------------------

def init_simple_training():

    config = TrainingConfig(
        n_qubits_bath=0,
        n_layers_gen=1,
        n_layers_disc=3,
        iterations=500,
        disc_steps=DISC_STEPS,
        gen_steps=1,
        lr_g=LR_G,
        lr_d=LR_D,
        batch_size=1, # Used data source has pure classes, thus no bath register and no randomness
        seed=100,
        cross_entropy_samples=200 # Small number for quick (but less accurate) testing
    )

    # Single class data source |+> = ( |0> + |1> ) / sqrt(2)
    ens_0 = QuantumEnsemble(
        states=[Statevector.from_label('+')],
        probs=[1.0]
    )
    data = QuantumDataSource(
        n_qubits_data=1,
        n_qubits_label=0,
        ensembles=[ens_0]
    )

    training = TrainQGAN(
        real_data=data,
        config=config
    )

    return training

def main():

    # config, training = init_cx_training()
    training = init_simple_training()
    print("Training initialized. Starting training...")

    result = training.run()

    # Final Verification
    print("\n--- Training Finished ---\n")

    print(f"Trained Generator Parameters:\n{result.gen_params}")
    print(f"Trained Discriminator Parameters:\n{result.disc_params}")

    # For each class, compute accurate (large sample) cross-entropy
    ces = compute_cross_entropy_over_labels_tensor(
        training.real_data,
        training.gen_sampler,
        result.gen_params,
        n_gen_samples_per_label=10000
    )

    print("\n--- Final Cross-Entropies per Class ---\n")
    for label, ce in enumerate(ces):
        print(f"Class {label}: Cross-Entropy = {ce:.6f}")


    # For each class, sample states from the trained Generator
    print("\n--- Generated Samples from Trained Generator ---")

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


