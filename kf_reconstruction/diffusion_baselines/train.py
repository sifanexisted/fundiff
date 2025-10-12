import os
import json
import time

import ml_collections
import wandb


import jax
from jax import random, vmap
import jax.numpy as jnp
from jax.experimental import mesh_utils, multihost_utils
from jax.sharding import Mesh, PartitionSpec as P

from function_diffusion.utils.model_utils import (
    create_optimizer,
    compute_total_params,
)
from function_diffusion.utils.checkpoint_utils import (
    create_checkpoint_manager,
    save_checkpoint,
)
from function_diffusion.utils.data_utils import create_dataloader
from function_diffusion.utils.dps_utils import create_step_fn, TrainState, Diffuser, get_burgers_res, create_train_state
from function_diffusion.models.cond_unet import Unet

from kf_reconstruction.data_utils import create_dataset


from dps_utils import get_div_res


def train_and_evaluate(config: ml_collections.ConfigDict):
    # Initialize model
    model = Unet(**config.model)
    # Create learning rate schedule and optimizer
    lr, tx = create_optimizer(config)

    # Create train state
    state = create_train_state(config, model, tx)
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

    def forward_fn(params, x, t, context=None):
        return state.apply_fn(params, x, t.flatten(), context=context)

    dfiffuser = Diffuser(forward_fn, get_burgers_res, config.ddpm)
    train_step = create_step_fn(dfiffuser, mesh)

    train_dataset, test_dataset = create_dataset(config)
    train_loader = create_dataloader(train_dataset,
                                     batch_size=config.dataset.train_batch_size,
                                     num_workers=config.dataset.num_workers)

    # Create checkpoint manager
    job_name = f"{config.model.model_name}_cond_{config.cond_diffusion}_use_pde_{config.use_pde_loss}"
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

    # Training loop
    rng_key = jax.random.PRNGKey(0)
    for epoch in range(10000):
        start_time = time.time()
        for x in train_loader:
            rng_key, subkey = jax.random.split(rng_key)

            x = jax.tree.map(jnp.array, x)
            if config.cond_diffusion:
                downsample_factors = jnp.array([1, 2, 4])
                random_downsample = jax.random.choice(subkey, downsample_factors)
                x_downsampled = x[:, ::random_downsample, ::random_downsample]
                context = jax.image.resize(x_downsampled, (x.shape[0], 256, 256, x.shape[-1]), method='bilinear')
            else:
                context = None

            x = jax.image.resize(x, (x.shape[0], 256, 256, x.shape[-1]), method='bilinear')
            batch = (x, context)
            batch = multihost_utils.host_local_array_to_global_array(
                batch, mesh, P("batch")
            )

            state, loss, rng_key = train_step(state, batch, rng_key)

        # Logging
        if epoch % config.logging.log_interval == 0:
            # Log metrics
            step = int(state.step)
            loss = loss.item()

            end_time = time.time()
            log_dict = {"loss": loss,
                        "lr": lr(step)}

            if jax.process_index() == 0:
                wandb.log(log_dict, step)  # Log metrics to W&B
                print("step: {}, loss: {:.3e},  time: {:.3e}".format(step, loss, end_time - start_time))

        # Save checkpoint
        if epoch % config.saving.save_interval == 0:
            save_checkpoint(ckpt_mngr, state)

        if step >= config.training.max_steps:
            break

    # Save final checkpoint
    print("Training finished, saving final checkpoint...")
    save_checkpoint(ckpt_mngr, state)
    ckpt_mngr.wait_until_finished()