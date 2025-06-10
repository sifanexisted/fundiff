from absl import app
from absl import flags
from ml_collections import config_flags

import jax
jax.config.update("jax_default_matmul_precision", "highest")

import train_autoencoder
import train_diffusion
import eval

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config",
    "configs/autoencoder.py",
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)


def main(argv):
    if FLAGS.config.mode == "train_autoencoder":
        train_autoencoder.train_and_evaluate(FLAGS.config)

    elif FLAGS.config.mode.startswith("train_diffusion"):
        train_diffusion.train_and_evaluate(FLAGS.config)

    elif FLAGS.config.mode == "eval":
        eval.evaluate(FLAGS.config)


if __name__ == "__main__":
    flags.mark_flags_as_required(["config"])
    app.run(main)
