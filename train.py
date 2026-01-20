from concurrent.futures import ThreadPoolExecutor
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

    # Architecture parameters
    max_workers: int = 10 # For parallel gradient computations

@dataclass
class TrainingResult:
    """
    Container for training results.
    """
    gen_params: torch.Tensor
    disc_params: torch.Tensor
    cross_entropies: list[float]
    expvals_RD: list[float]
    expvals_GD: list[float]


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
        # Discriminator: Initialize widely [-pi/2, pi/2] to avoid barren plateaus.
        disc_w = torch.nn.Parameter(
            torch.tensor(np.random.uniform(-np.pi/2, np.pi/2, self.qgan.n_disc_params))
        )

        # Use Adam optimizers for both Generator and Discriminator
        opt_g = torch.optim.Adam([gen_w], lr=self.config.lr_g)
        opt_d = torch.optim.Adam([disc_w], lr=self.config.lr_d)

        # Track losses and cross-entropy over iterations
        cross_entropies = []
        expvals_RD = []
        expvals_GD = []

        # Main training loop
        for iteration in range(self.config.iterations):

            # A) Train Discriminator
            for _ in range(self.config.disc_steps):
                
                opt_d.zero_grad()

                # Compute discriminator gradient step
                grad_d, expval_RD, expval_GD = self.discriminator_step(disc_w=disc_w, gen_w=gen_w)

                # Apply optimizer step for Discriminator
                disc_w.grad = grad_d / 4.0

                opt_d.step()

                expvals_RD.append(expval_RD)
                expvals_GD.append(expval_GD)


            # Optional tracking: save Discriminator loss after multi-step update


            # B) Train Generator
            for _ in range(self.config.gen_steps):

                opt_g.zero_grad()

                # Compute generator gradient step
                grad_g, expval_GD = self.generator_step(gen_w=gen_w, disc_w=disc_w)

                # Apply optimizer step for Generator
                gen_w.grad = grad_g / 4.0

                opt_g.step()

                expvals_GD.append(expval_GD)
                expvals_RD.append(expvals_RD[-1])  # Last RD value remains the same
            
            # Optional tracking: save Generator loss after multi-step update

            # Optional tracking: compute and save cross entropy after generator update
            cross_entropies.append(compute_cross_entropy_over_labels_tensor(
                self.real_data,
                self.gen_sampler,
                gen_w,
                n_gen_samples_per_label=self.config.cross_entropy_samples,
                eps=1e-12,
            ))

            # Lower learning rates after iteration 200
            if iteration == 150:
                for g in opt_g.param_groups:
                    g["lr"] *= 0.5
                for g in opt_d.param_groups:
                    g["lr"] *= 0.5

            # Print progress every 5 iterations
            if iteration % 10 == 0:
                print(f"Iteration {iteration:03d}", end="")
                for label in range(self.real_data.num_classes):
                    print(f" | CE_{label}: {cross_entropies[-1][label]:.4f}", end="")
                print("")
                # Print expvals for monitoring
                print(f"    Expval_RD: {expvals_RD[-1]:.4f}, Expval_GD: {expvals_GD[-1]:.4f}")
                print(f"    Total Loss: {expvals_RD[-1] - expvals_GD[-1]:.4f}")
                print("")

            # Print training checkpoint
            if iteration % 100 == 0:
                print(f"\n--- Training checkpoint (it={iteration}) ---")
                print(f"cross-entropies: {cross_entropies}")
                print(f"expvals_RD: {expvals_RD}")
                print(f"expvals_GD: {expvals_GD}")


        # Return trained parameters and any tracked metrics
        result = TrainingResult(
            gen_params=gen_w.detach(),
            disc_params=disc_w.detach(),
            cross_entropies=cross_entropies,
            expvals_RD=expvals_RD,
            expvals_GD=expvals_GD,
        )
        return result


    def discriminator_step(
        self,
        disc_w: torch.Tensor,
        gen_w: torch.Tensor
    ):
        # Lambda to build args for gradient computation tasks
        def build_args(label):
            return (label, disc_w, gen_w, self.qgan)

        # Run classes x batches tasks in parallel for a single gradient step
        results = self.run_parallel_gradient_computation(
            self.disc_batch_task,
            build_args
        )

        # Accumulate gradients and expectation values
        grad_d = torch.zeros_like(disc_w)
        expval_RD_total = 0.0
        expval_GD_total = 0.0

        for grad_RD, grad_GD, expval_RD, expval_GD in results:
            expval_RD_total += expval_RD
            expval_GD_total += expval_GD

            # Formula for loss is: Loss = expval_RD - expval_GD
            # Discriminator tries to maximize this quantity
            # Formula for gradient is thus: Grad_G = -(grad_RD - grad_GD), since pyTorch does gradient descent
            grad_d += grad_GD - grad_RD

        # Normalize gradients
        normalizer = self.real_data.num_classes * self.config.batch_size
        grad_d /= normalizer

        return grad_d, expval_RD_total / normalizer, expval_GD_total / normalizer


    def generator_step(
        self,
        gen_w: torch.Tensor,
        disc_w: torch.Tensor,
    ):
        # Lambda to build args for gradient computation tasks
        def build_args(label):
            return (label, gen_w, disc_w, self.qgan)

        # Run classes x batches tasks in parallel for a single gradient step
        results = self.run_parallel_gradient_computation(
            self.gen_batch_task,
            build_args
        )

        # Accumulate gradients and expectation values
        grad_g = torch.zeros_like(gen_w)
        expval_GD_total = 0.0

        for grad_GD, expval_GD in results:
            expval_GD_total += expval_GD

            # Formula for loss is: Loss = expval_RD - expval_GD
            # Generator tries to minimize this quantity
            # Formula for gradient is thus: Grad_G = - grad_GD, since pyTorch does gradient descent
            grad_g += -grad_GD

        # Normalize gradients
        normalizer = self.real_data.num_classes * self.config.batch_size
        grad_g /= normalizer

        return grad_g, expval_GD_total / normalizer


    @staticmethod
    def disc_batch_task(args) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        """
        One discriminator task for a single (label, batch) sample.
        """
        label, disc_w_np, gen_w_np, qgan = args

        grad_RD, expval_RD = Gradients.RD_disc_grad(
            disc_w_np, qgan, label
        )
        grad_GD, expval_GD = Gradients.GD_disc_grad(
            disc_w_np, gen_w_np, qgan, label
        )

        return grad_RD, grad_GD, expval_RD, expval_GD


    @staticmethod
    def gen_batch_task(args) -> tuple[torch.Tensor, float]:
        """
        One generator task for a single (label, batch) sample.
        """
        label, gen_w_np, disc_w_np, qgan = args

        grad_GD, expval_GD = Gradients.GD_gen_grad(
            gen_w_np, disc_w_np, qgan, label
        )

        return grad_GD, expval_GD

    
    def run_parallel_gradient_computation(
        self,
        task_fn,
        task_args_builder
    ):
        """
        Generic parallel executor over (label, batch) tasks.
        """
        tasks = []
        for label in range(self.real_data.num_classes):
            for _ in range(self.config.batch_size):
                tasks.append(task_args_builder(label))

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            return list(executor.map(task_fn, tasks))


