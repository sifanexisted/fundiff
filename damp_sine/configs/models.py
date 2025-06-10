import ml_collections


MODEL_CONFIGS = {}


def _register(get_config):
    """Adds reference to model config into MODEL_CONFIGS."""
    config = get_config().lock()
    name = config.get("model_name")
    MODEL_CONFIGS[name] = config
    return get_config


@_register
def get_fae_config():
    config = ml_collections.ConfigDict()
    config.model_name = "FAE"   # Function Autoencoder

    config.encoder = encoder = ml_collections.ConfigDict()
    encoder.grid_size = (100,)
    encoder.patch_size = (4,)
    encoder.num_latents = 100
    encoder.emb_dim = 256
    encoder.depth = 6
    encoder.num_heads = 8
    encoder.mlp_ratio = 1
    encoder.layer_norm_eps = 1e-5

    config.decoder = decoder = ml_collections.ConfigDict()
    decoder.fourier_freq = 1.0
    decoder.dec_emb_dim = 256
    decoder.dec_depth = 2
    decoder.dec_num_heads = 8
    decoder.num_mlp_layers = 2
    decoder.mlp_ratio = 1
    decoder.out_dim = 1
    decoder.layer_norm_eps = 1e-5

    return config


@_register
def get_dit_config():
    config = ml_collections.ConfigDict()
    config.model_name = "DiT"

    config.grid_size = (100,)   # equal to num_latents
    config.emb_dim = 256
    config.depth = 6
    config.num_heads = 8
    config.mlp_ratio = 1
    config.out_dim = 256

    return config

