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
    config.model_name = "FAE"

    config.encoder = encoder = ml_collections.ConfigDict()
    encoder.patch_size = (16, 16)   # For tokenization
    encoder.emb_dim = 256
    encoder.num_latents = 256
    encoder.grid_size = (256, 256)   # Maximum resolution for positional embeddings (H / P, W / P)
    encoder.depth = 8
    encoder.num_heads = 8
    encoder.mlp_ratio = 2
    encoder.layer_norm_eps = 1e-5

    config.decoder = decoder = ml_collections.ConfigDict()
    decoder.period = True
    decoder.fourier_freq = 5.0
    decoder.dec_emb_dim = 256
    decoder.dec_depth = 4
    decoder.dec_num_heads = 8
    decoder.mlp_ratio = 2
    decoder.num_mlp_layers = 2
    decoder.out_dim = 2
    decoder.layer_norm_eps = 1e-5

    return config


@_register
def get_dit_config():
    config = ml_collections.ConfigDict()
    config.model_name = "DiT"

    config.emb_dim = 256
    config.depth = 8
    config.num_heads = 16
    config.mlp_ratio = 2
    config.out_dim = 256

    return config


@_register
def get_cvit_config():
    config = ml_collections.ConfigDict()
    config.model_name = "CViT"

    config.encoder = encoder = ml_collections.ConfigDict()

    encoder.patch_size = (16, 16)
    encoder.grid_size = (256, 256)
    encoder.emb_dim = 256
    encoder.depth = 8
    encoder.num_heads = 8
    encoder.mlp_ratio = 2
    encoder.layer_norm_eps = 1e-5

    config.decoder = decoder = ml_collections.ConfigDict()

    decoder.period = None
    decoder.fourier_freq = 1.0
    decoder.dec_emb_dim = 256
    decoder.dec_depth = 4
    decoder.mlp_ratio = 2
    decoder.dec_num_heads = 8
    decoder.num_mlp_layers = 2
    decoder.out_dim = 2

    return config


