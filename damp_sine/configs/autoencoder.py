import ml_collections

from configs import models


def get_config(model):
    """Get the hyperparameter configuration for a specific model."""
    config = get_base_config()
    get_model_config = getattr(models, f"get_{model}_config")
    config.model = get_model_config()
    return config


def get_base_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    # Random seed
    config.seed = 42

    # Input shape for initializing Flax models
    config.x_dim = [8, 128, 2]
    config.coords_dim = [1,]  # Only for initializing CViT model

    # Training or evaluation
    config.mode = "train_autoencoder"

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "fundiff_damp_sine"
    wandb.tag = None

    # Dataset
    config.dataset = dataset = ml_collections.ConfigDict()
    dataset.num_samples = 1024
    dataset.num_sensors = 128
    dataset.downsample_factor = 1
    dataset.train_batch_size = 256  # Per device
    dataset.test_batch_size = 256  # Per device
    dataset.num_workers = 8

    # Learning rate
    config.lr = lr = ml_collections.ConfigDict()
    lr.init_value = 0.0
    lr.peak_value = 1e-3
    lr.decay_rate = 0.9
    lr.transition_steps = 1000
    lr.warmup_steps = 1000

    # Optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.weight_decay = 1e-5
    optim.clip_norm = 1.0

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 5 * 10**4
    training.num_queries = 128
    training.random_sensors = False
    training.sensor_range = [64, 128]

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_interval = 1000

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_interval = 1000
    saving.num_keep_ckpts = 2

    return config
