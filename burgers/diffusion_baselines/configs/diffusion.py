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
    config.x_dim = [2, 256, 256, 1]
    # config.context_dim = [2, 128, 128, 1]

    # Training or evaluation
    config.mode = "train_ddpm"  # options: train_ve | train_ddpm | train_edm

    # physics-informed training
    config.use_pde_loss = False
    config.cond_diffusion = False

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "fundiff_burgers"
    wandb.tag = None

    # Dataset
    config.dataset = dataset = ml_collections.ConfigDict()
    dataset.data_path = "/scratch/sifanw/transformer_as_integrator/burgers/burger_nu_1e-3.mat"
    dataset.downsample_factor = 1
    dataset.num_train_samples = 3600
    dataset.train_batch_size = 16  # Per device
    dataset.test_batch_size = 4  # Per device
    dataset.num_workers = 8

    # ddpm
    config.ddpm = ddpm = ml_collections.ConfigDict()
    ddpm.beta_1 = 1e-4
    ddpm.beta_T = 0.02
    ddpm.T = 1000
    ddpm.schedule = 'linear' # options: linear, cosine, sigmoid


    # Learning rate
    config.lr = lr = ml_collections.ConfigDict()
    lr.init_value = 0.0
    lr.peak_value = 1e-4
    lr.decay_rate = 0.9
    lr.transition_steps = 2000
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

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_interval = 5

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_interval = 5
    saving.num_keep_ckpts = 2

    return config
