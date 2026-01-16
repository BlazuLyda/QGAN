import torch
import numpy as np
from dataclasses import dataclass
from circuits import QGANCircuits
from data import QuantumDataSource
from gradients import RealDiscExpval, GenDiscExpval


@dataclass
class TrainingConfig:
    """
    Configuration parameters for training the QGAN.
    """
    # Circuit parameters
    n_qubits_bath: int = 0
    n_layers_gen: int = 2 # This should be tuned per dataset
    n_layers_disc: int = 4 # This should be tuned per dataset

    # Training parameters
    iterations: int = 1000
    gen_steps: int = 1
    disc_steps: int = 5
    lr_g: float = 0.01
    lr_d: float = 0.01
    batch_size: int = 32 # For use later if mini-batching is implemented
    seed: int|None = None


class TrainQGAN:

    def __init__(self, real_data: QuantumDataSource, config: TrainingConfig) -> None:

        self.real_data = real_data
        self.config = config  
        self.rng = np.random.default_rng(config.seed)

        self.qgan = QGANCircuits(
            real_source=real_data,
            n_data_qubits=real_data.n_data,
            n_label_qubits=real_data.n_label,
            n_bath_qubits=config.n_qubits_bath,
            n_layers_gen=config.n_layers_gen,
            n_layers_disc=config.n_layers_disc,
        )


    def run(self):
        """
        Main training loop for the QGAN.
        """


        # Initialize Generator and Discriminator parameters
        # Generator: Initialize closer to 0 to preserve Label early on.
        gen_w = torch.nn.Parameter(
            torch.tensor(np.random.uniform(-0.1, 0.1, self.qgan.n_gen_params))
        )
        # Discriminator: Initialize widely [-pi, pi] to avoid barren plateaus.
        disc_w = torch.nn.Parameter(
            torch.tensor(np.random.uniform(-np.pi, np.pi, self.qgan.n_disc_params))
        )

        # Use Adam optimizers for both Generator and Discriminator
        opt_g = torch.optim.Adam([gen_w], lr=self.config.lr_g)
        opt_d = torch.optim.Adam([disc_w], lr=self.config.lr_d)


        for iteration in range(self.config.iterations):

            # A) Train Discriminator
            for _ in range(self.config.disc_steps):
                opt_d.zero_grad()
                total_loss_d = 0

                for label in range(self.real_data.num_classes):
                    for _ in range(self.config.batch_size):
                        # Comopute partial gradients of DR circuit in relation to disc_w
                        exp_real = RealDiscExpval.apply(
                            disc_w, self.qgan, label
                        )
                        # Compute partial gradients of DG circuit in relation to disc_w
                        # TODO: add rng
                        exp_fake = GenDiscExpval.apply(
                            gen_w.detach(), disc_w, self.qgan, label
                        )

                        # Minimax Loss D: -( E[Real] - E[Fake] )
                        # Ideally converges to -2 (if D is perfect and G is bad) or 0 (if G is perfect)
                        loss_d_label = -(exp_real - exp_fake)
                        total_loss_d += loss_d_label

                # Apply optimizer step for Discriminator
                total_loss_d.backward()
                opt_d.step()
                # Optional tracking: save Discriminator loss after update

            # B) Train Generator
            for _ in range(self.config.gen_steps):
                opt_g.zero_grad()
                total_loss_g = 0

                for label in range(self.real_data.num_classes):
                    for _ in range(self.config.batch_size):
                        # Compute partial gradients of DG circuit in relation to gen_w
                        # TODO: add rng
                        exp_fake = GenDiscExpval.apply(gen_w, disc_w.detach(), self.qgan, label)

                        # Minimax Loss G: - E[Fake]
                        loss_g_label = -exp_fake
                        total_loss_g += loss_g_label

                # Apply optimizer step for Generator
                total_loss_g.backward()
                opt_g.step()
                # Optional tracking: save Generator loss after update

            
            # Optional tracking: compute and save cross entropy after generator update
