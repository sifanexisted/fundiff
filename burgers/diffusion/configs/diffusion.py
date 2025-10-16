import ml_collections

from configs import models


def get_config(autoencoder_diffusion):
    """Get the hyperparameter configuration for a specific model."""
    config = get_base_config()

    autoencoder, diffusion = autoencoder_diffusion.split(',')

    get_autoencoder_config = getattr(models, f"get_{autoencoder}_config")
    get_diffusion_config = getattr(models, f"get_{diffusion}_config")

    config.autoencoder = get_autoencoder_config()
    config.diffusion = get_diffusion_config()
    return config


def get_base_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    # Random seed
    config.seed = 42

    # Input shape for initializing Flax models
    config.x_dim = [2, 200, 200, 1]
    config.z_dim = [2, 100, 256]
    config.c_dim = [2, 100, 256]
    config.t_dim = [2,]
    config.coords_dim = [2,]  # Only for initializing CViT model

    # Training or evaluation
    config.mode = "train_diffusion"

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "fundiff_burgers"
    wandb.tag = None

    # Dataset
    config.dataset = dataset = ml_collections.ConfigDict()
    dataset.data_path = "/scratch/sifanw/transformer_as_integrator/burgers/burger_nu_1e-3.mat"
    dataset.downsample_factor = 1
    dataset.batch_size = 128  # Per device
    dataset.num_train_samples = 3600
    dataset.test_batch_size = 4  # Per device
    dataset.num_workers = 8

    # Learning rate
    config.lr = lr = ml_collections.ConfigDict()
    lr.init_value = 0.0
    lr.peak_value = 1e-3
    lr.decay_rate = 0.9
    lr.transition_steps = 5000
    lr.warmup_steps = 2000

    # Optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.weight_decay = 1e-5
    optim.clip_norm = 1.0

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 1 * 10**5
    training.num_queries = 256
    training.random_resolution = True
    training.use_pde = False

    # Evaluation
    config.eval = eval = ml_collections.ConfigDict()
    eval.num_samples = 1024
    eval.num_steps = 100
    eval.batch_size = 64

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_interval = 20

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_interval = 20
    saving.num_keep_ckpts = 2

    return config
