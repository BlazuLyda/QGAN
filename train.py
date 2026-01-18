import torch
import numpy as np
from dataclasses import dataclass
from circuits import GenCircuit, QGANCircuits
from data import QuantumDataSource
from evaluation import compute_cross_entropy_over_labels_tensor
from gradients import Gradients

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

    # Logging parameters
    cross_entropy_samples: int = 2000 # Number of samples to estimate cross-entropy

@dataclass
class TrainingResult:
    """
    Container for training results.
    """
    gen_params: torch.Tensor
    disc_params: torch.Tensor
    cross_entropies: list[float]


class TrainQGAN:

    def __init__(self, real_data: QuantumDataSource, config: TrainingConfig) -> None:

        self.real_data = real_data
        self.config = config  

        self.qgan = QGANCircuits(
            real_source=real_data,
            n_data_qubits=real_data.n_data,
            n_label_qubits=real_data.n_label,
            n_bath_qubits=config.n_qubits_bath,
            n_layers_gen=config.n_layers_gen,
            n_layers_disc=config.n_layers_disc,
            random=config.seed
        )
        
        self.gen_sampler = GenCircuit(
            n_data_qubits=real_data.n_data,
            n_label_qubits=real_data.n_label,
            n_bath_qubits=config.n_qubits_bath,
            n_layers=config.n_layers_gen,
        )


    def run(self):
        """
        Main training loop for the QGAN.
        """

        # Setup pytorch environment
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(self.config.seed)

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

        # Track losses and cross-entropy over iterations
        cross_entropies = []

        # Main training loop
        for iteration in range(self.config.iterations):

            # A) Train Discriminator
            for _ in range(self.config.disc_steps):
                opt_d.zero_grad()

                # Accumulate expectation values over all classes and batch (for tracking loss)
                expval_RD_total = 0.0
                expval_GD_total = 0.0

                # Accumulate gradients over all classes and batch
                grad_d = torch.zeros_like(disc_w)

                # Iterate over all classes
                for label in range(self.real_data.num_classes):
                    for _ in range(self.config.batch_size):

                        # Compute single-point gradients of RD and GD circuits
                        grad_RD, expval_RD = Gradients.RD_disc_grad(disc_w, self.qgan, label)
                        grad_GD, expval_GD = Gradients.GD_disc_grad(disc_w, gen_w, self.qgan, label)

                        # Formula for loss is: Loss = expval_RD - expval_GD
                        # Discriminator tries to maximize this quantity
                        # Formula for gradient is thus: Grad_G = -(grad_RD - grad_GD), since pyTorch does gradient descent
                        expval_RD_total += expval_RD
                        expval_GD_total += expval_GD
                        grad_d += grad_GD - grad_RD

                # Average gradients over batch and classes
                grad_d /= (self.real_data.num_classes * self.config.batch_size)

                # Apply optimizer step for Discriminator
                disc_w.grad = grad_d
                opt_d.step()

            # Optional tracking: save Discriminator loss after multi-step update


            # B) Train Generator
            for _ in range(self.config.gen_steps):
                opt_g.zero_grad()
                grad_g = torch.zeros_like(gen_w)

                # Iterate over all classes
                for label in range(self.real_data.num_classes):
                    for _ in range(self.config.batch_size):

                        # Compute single-point gradient of GD circuit
                        grad_GD, expval_GD = Gradients.GD_gen_grad(gen_w, disc_w, self.qgan, label)

                        # Formula for loss is: Loss = expval_RD - expval_GD
                        # Generator tries to minimize this quantity
                        # Formula for gradient is thus: Grad_G = - grad_GD, since pyTorch does gradient descent
                        grad_g += -grad_GD

                # Average gradients over batch and classes
                grad_g /= (self.real_data.num_classes * self.config.batch_size)

                # Apply optimizer step for Generator
                gen_w.grad = grad_g
                opt_g.step()

            
            # Optional tracking: save Generator loss after multi-step update

            # Optional tracking: compute and save cross entropy after generator update
            cross_entropies.append(compute_cross_entropy_over_labels_tensor(
                self.real_data,
                self.gen_sampler,
                gen_w,
                n_gen_samples_per_label=self.config.cross_entropy_samples,
                eps=1e-12,
            ))

            # Print progress every 5 iterations
            if iteration % 5 == 0:
                print(f"Iteration {iteration:03d}", end="")
                for label in range(self.real_data.num_classes):
                    print(f" | CE_{label}: {cross_entropies[-1][label]:.4f}", end="")
                print("")


        # Return trained parameters and any tracked metrics
        result = TrainingResult(
            gen_params=gen_w.detach(),
            disc_params=disc_w.detach(),
            cross_entropies=cross_entropies
        )
        return result
