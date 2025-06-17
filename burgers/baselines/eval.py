import os

import ml_collections


import jax
import jax.numpy as jnp
from jax import vmap

from jax.experimental import mesh_utils, multihost_utils
from jax.sharding import Mesh, PartitionSpec as P

from function_diffusion.utils.model_utils import (
    create_model,
    create_train_state,
    create_optimizer,
    compute_total_params,
)
from function_diffusion.utils.checkpoint_utils import (
    create_checkpoint_manager,
    restore_checkpoint,
)
from function_diffusion.utils.data_utils import create_dataloader
from function_diffusion.utils.baseline_utils import create_eval_step

from data_utils import create_dataset

def evaluate(config: ml_collections.ConfigDict):
    # Initialize model
    model = create_model(config)
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

    # Create checkpoint manager
    job_name = f"{config.model.model_name}_downsample_{config.dataset.downsample_factor}"
    ckpt_path = os.path.join(os.getcwd(), job_name, "ckpt")
    ckpt_mngr = create_checkpoint_manager(config.saving, ckpt_path)

    state = restore_checkpoint(ckpt_mngr, state)
    print(f"Model loaded from step {state.step}")

    # Create sharding for data parallelism
    mesh = Mesh(mesh_utils.create_device_mesh((jax.device_count(),)), "batch")
    state = multihost_utils.host_local_array_to_global_array(state, mesh, P())

    # Create loss and train step functions
    # train_step = create_train_step(model, mesh)
    eval_step = create_eval_step(model, mesh)

    # Create dataloaders
    train_dataset, test_dataset = create_dataset(config)
    test_loader = create_dataloader(test_dataset,
                                    batch_size=config.dataset.train_batch_size,
                                    num_workers=config.dataset.num_workers)

    def compute_error(pred, y):
        return jnp.linalg.norm(pred.flatten() - y.flatten()) / jnp.linalg.norm(y.flatten())

    downsample_factor = 2

    error_list = []

    rng_key = jax.random.PRNGKey(0)

    for x in test_loader:
        rng_key, subkey = jax.random.split(rng_key)
        x = jax.tree.map(jnp.array, x)
        y = x
        x_downsampled = x[:, ::downsample_factor, ::downsample_factor]

        noise = jax.random.normal(rng_key, x_downsampled.shape) * 0.2 * 0.10
        x_noise = x_downsampled + noise

        x = jax.image.resize(x_noise, (x.shape[0], 256, 256, x.shape[-1]), method='bilinear')
        y = jax.image.resize(y, x.shape, method='bilinear')
        batch = (x, y)

        pred = eval_step(state.params, batch)

        error = vmap(compute_error)(pred, y)

        error_list.append(error)

    return None
