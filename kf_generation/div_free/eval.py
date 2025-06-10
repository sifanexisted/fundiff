import os

from tqdm import tqdm

import ml_collections
import numpy as np

import jax
import jax.numpy as jnp


from jax.experimental import mesh_utils, multihost_utils
from jax.sharding import Mesh, PartitionSpec as P

from function_diffusion.models import Encoder, Decoder

from function_diffusion.utils.model_utils import (
    create_autoencoder_state,
    create_diffusion_state,
    create_optimizer,
    compute_total_params,
)
from function_diffusion.utils.train_utils import create_train_diffusion_step, get_diffusion_batch
from function_diffusion.utils.checkpoint_utils import (
    create_checkpoint_manager,
    save_checkpoint,
    restore_checkpoint,
)

from model_utils import create_decoder_step

def evaluate(config: ml_collections.ConfigDict):
    # Initialize function autoencoder
    encoder = Encoder(**config.autoencoder.encoder)
    decoder = Decoder(**config.autoencoder.decoder)

    # Create learning rate schedule and optimizer
    lr, tx = create_optimizer(config)

    # Create train state
    state = create_autoencoder_state(config, encoder, decoder, tx)

    # Create checkpoint manager
    fae_job_name = f"{config.autoencoder.model_name}_downsample_{config.dataset.downsample_factor}"

    ckpt_path = os.path.join(os.getcwd(), fae_job_name, "ckpt")
    ckpt_mngr = create_checkpoint_manager(config.saving, ckpt_path)

    # Restore the model from the checkpoint
    print(f"Restored model {fae_job_name} from step", state.step)
    state = restore_checkpoint(ckpt_mngr, state)

    # Create sharding for data parallelism
    mesh = Mesh(mesh_utils.create_device_mesh((jax.device_count(),)), "batch")
    state = multihost_utils.host_local_array_to_global_array(state, mesh, P())

    # Load generated latent samples
    dit_job_name = f"{config.diffusion.model_name}_downsample_{config.dataset.downsample_factor}"
    data_path = os.path.join(os.getcwd(), dit_job_name, "z1.npy")
    z1_list = np.load(data_path)
    batch_size = config.eval.batch_size

    decoder_step = create_decoder_step(decoder, mesh)

    # Generate samples
    h, w = config.eval.resolution

    x_coords = jnp.linspace(0, 1, w)
    y_coords = jnp.linspace(0, 1, h)
    x_coords, y_coords = jnp.meshgrid(x_coords, y_coords, indexing='ij')
    coords = jnp.hstack([x_coords.reshape(-1, 1), y_coords.reshape(-1, 1)])
    coords = multihost_utils.host_local_array_to_global_array(coords, mesh, P())

    u_preds_list = []

    for i in tqdm(range(z1_list.shape[0] // batch_size)):
        # Generate a new random seed for each sample
        new_z1 = z1_list[batch_size * i:batch_size * (i + 1)]
        new_z1 = jnp.array(new_z1)

        # Generate predictions for the sample
        u_pred = decoder_step(state.params[1], new_z1, coords)
        u_pred = u_pred.reshape(-1, h, w, 3)
        # Append the predictions to the lists
        u_preds_list.append(np.array(u_pred))


    # Save the generated samples
    u_preds_list = np.concatenate(u_preds_list, axis=0)

    save_path = os.path.join(os.getcwd(), dit_job_name)
    np.save(os.path.join(save_path, "pred_list.npy"), u_preds_list)
    print('Successfully saved the generated samples to', save_path)

    return None
