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

    # Training or evaluation
    config.mode = "train_ddpm"  # options: train_ve | train_ddpm | train_edm

    # physics-informed training
    config.use_pde_loss = False
    # config.physics_informed_sample = False

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "fundiff_burgers"
    wandb.tag = None

    # Dataset
    config.dataset = dataset = ml_collections.ConfigDict()
    dataset.data_path = "/scratch/sifanw/transformer_as_integrator/burgers/burger_nu_1e-3.mat"
    dataset.downsample_factor = 1
    dataset.num_train_samples = 3600
    dataset.train_batch_size = 32  # Per device
    dataset.test_batch_size = 4  # Per device
    dataset.num_workers = 8

    # ddpm
    config.ddpm = ddpm = ml_collections.ConfigDict()
    ddpm.beta_schedule = 'cosine'
    ddpm.timesteps = 1000
    ddpm.p2_loss_weight_gamma = 0.  # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
    ddpm.p2_loss_weight_k = 1
    ddpm.self_condition = True  # not tested yet
    ddpm.is_pred_x0 = True  # by default, the model will predict noise, if True predict x0

    # ve
    config.ve = ve = ml_collections.ConfigDict()
    ve.sigma_min = 8e-2
    ve.sigma_max = 80.

    # DiffusionPDE
    # sigma_min: 0.002
    # sigma_max: 80

    # edm
    config.edm = edm = ml_collections.ConfigDict()
    edm.timesteps = 1000
    edm.P_mean = -1.2
    edm.P_std = 1.2
    edm.sigma_data = 0.5
    edm.is_pred_x0 = False
    edm.sigma_min = 0.002
    edm.sigma_max = 80.0
    edm.rho = 7

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
