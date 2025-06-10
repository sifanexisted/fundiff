import os

import jax
from jax.experimental import mesh_utils, multihost_utils

import orbax.checkpoint as ocp

from function_diffusion.utils.model_utils import create_optimizer, create_autoencoder_state


def create_checkpoint_manager(config, ckpt_path):
    multihost_utils.sync_global_devices("before_ckpt_mngr")
    ckpt_options = ocp.CheckpointManagerOptions(max_to_keep=config.num_keep_ckpts)
    if not os.path.isdir(ckpt_path):
        os.makedirs(ckpt_path)
    ckpt_mngr = ocp.CheckpointManager(ckpt_path, options=ckpt_options)
    return ckpt_mngr


def save_checkpoint(ckpt_mngr, state):
    multihost_utils.sync_global_devices("before_ckpt_save")
    with jax.spmd_mode("allow_all"):
        ckpt_mngr.save(state.step, args=ocp.args.StandardSave(state))


def restore_checkpoint(ckpt_mngr, state):
    multihost_utils.sync_global_devices("before_ckpt_restore")
    with jax.spmd_mode("allow_all"):
        state = ckpt_mngr.restore(
            ckpt_mngr.latest_step(),
            args=ocp.args.StandardRestore(state),
        )
    return state


def restore_fae_state(config, job_name, encoder, decoder):
    # Create learning rate schedule and optimizer
    lr, tx = create_optimizer(config)

    # Create train state
    state = create_autoencoder_state(config, encoder, decoder, tx)

    # Create checkpoint manager
    ckpt_path = os.path.join(os.getcwd(), job_name, "ckpt")
    ckpt_mngr = create_checkpoint_manager(config.saving, ckpt_path)

    # Restore the model from the checkpoint
    fae_state = restore_checkpoint(ckpt_mngr, state)
    print(f"Restored model {job_name} from step", fae_state.step)

    return fae_state