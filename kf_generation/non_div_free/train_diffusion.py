import os
import json
import time

import ml_collections
import wandb

import jax
import jax.numpy as jnp
from jax import random, jit

from jax.experimental import mesh_utils, multihost_utils
from jax.sharding import Mesh, PartitionSpec as P

from function_diffusion.models import DiT, Encoder, Decoder

from function_diffusion.utils.model_utils import (
    create_optimizer,
    create_autoencoder_state,
    create_diffusion_state,
    compute_total_params,
)
from function_diffusion.utils.train_utils import create_train_diffusion_step, get_diffusion_batch, sample_ode
from function_diffusion.utils.checkpoint_utils import (
    create_checkpoint_manager,
    save_checkpoint,
    restore_fae_state
)
from function_diffusion.utils.data_utils import create_dataloader
from kf_generation.data_utils import create_dataset
from model_utils import create_encoder_step


def train_and_evaluate(config: ml_collections.ConfigDict):
    # Initialize function autoencoder
    encoder = Encoder(**config.autoencoder.encoder)
    decoder = Decoder(**config.autoencoder.decoder)
    fae_job_name = f"{config.autoencoder.model_name}"
    fae_state = restore_fae_state(config, fae_job_name, encoder, decoder)

    # Initialize diffusion model
    dit = DiT(**config.diffusion)
    # Create learning rate schedule and optimizer
    lr, tx = create_optimizer(config)

    # Create diffusion train state
    state = create_diffusion_state(config, dit, tx)
    num_params = compute_total_params(state)
    print(f"Model storage cost: {num_params * 4 / 1024 / 1024:.2f} MB of parameters")

    # Device count
    num_local_devices = jax.local_device_count()
    num_devices = jax.device_count()
    print(f"Number of devices: {num_devices}")
    print(f"Number of local devices: {num_local_devices}")

    # Create sharding for data parallelism
    mesh = Mesh(mesh_utils.create_device_mesh((jax.device_count(),)), "batch")
    state = multihost_utils.host_local_array_to_global_array(state, mesh, P())
    fae_state = multihost_utils.host_local_array_to_global_array(fae_state, mesh, P())

    # Create train step function
    train_step = create_train_diffusion_step(dit, mesh)
    encoder_step = create_encoder_step(encoder, mesh)

    # Create dataloaders
    train_dataset, _ = create_dataset(config)
    train_loader = create_dataloader(train_dataset,
                                     batch_size=config.dataset.batch_size,
                                     num_workers=config.dataset.num_workers)

    # Create checkpoint manager
    job_name = f"{config.diffusion.model_name}"
    ckpt_path = os.path.join(os.getcwd(), job_name, "ckpt")
    if jax.process_index() == 0:
        if not os.path.isdir(ckpt_path):
            os.makedirs(ckpt_path)

        # Save config
        config_dict = config.to_dict()
        config_path = os.path.join(os.getcwd(), job_name, "config.json")
        with open(config_path, "w") as json_file:
            json.dump(config_dict, json_file, indent=4)

        # Initialize W&B
        wandb_config = config.wandb
        wandb.init(project=wandb_config.project, name=job_name, config=config)

    # Create checkpoint manager
    ckpt_mngr = create_checkpoint_manager(config.saving, ckpt_path)

    rng = random.PRNGKey(0)
    for epoch in range(10000):
        start_time = time.time()
        for batch in train_loader:
            rng, _ = random.split(rng)
            batch = jax.tree.map(jnp.array, batch)
            u = batch
            u_batch = (jnp.ones_like(u), u, jnp.ones_like(u))

            # # Shard the batch across devices
            u_batch = multihost_utils.host_local_array_to_global_array(
                u_batch, mesh, P("batch")
                )

            z_u = encoder_step(fae_state.params[0], u_batch)

            batch, rng = get_diffusion_batch(rng, z1=z_u)
            state, loss = train_step(state, batch)

        # Logging
        if epoch % config.logging.log_interval == 0:
            # Log metrics
            step = int(state.step)
            loss = loss.item()
            end_time = time.time()
            log_dict = {"loss": loss, "lr": lr(step)}

            if jax.process_index() == 0:
                wandb.log(log_dict, step)  # Log metrics to W&B
                print("step: {}, loss: {:.3e}, time: {:.3e}".format(step, loss, end_time - start_time))

        # Save checkpoint
        if epoch % config.saving.save_interval == 0:
            save_checkpoint(ckpt_mngr, state)

        if step >= config.training.max_steps:
            break

    # Save final checkpoint
    print("Training finished, saving final checkpoint...")
    save_checkpoint(ckpt_mngr, state)
    ckpt_mngr.wait_until_finished()




