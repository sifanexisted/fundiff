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
    encoder.patch_size = (10, 5)   # For tokenization
    encoder.emb_dim = 256
    encoder.num_latents = 256
    encoder.grid_size = (200, 100)   # Maximum resolution for positional embeddings (H / P, W / P)
    encoder.depth = 8
    encoder.num_heads = 8
    encoder.mlp_ratio = 2
    encoder.layer_norm_eps = 1e-5

    config.decoder = decoder = ml_collections.ConfigDict()
    decoder.period = False
    decoder.fourier_freq = 1.0
    decoder.dec_emb_dim = 256
    decoder.dec_depth = 4
    decoder.dec_num_heads = 8
    decoder.mlp_ratio = 2
    decoder.num_mlp_layers = 2
    decoder.out_dim = 1
    decoder.layer_norm_eps = 1e-5

    return config


@_register
def get_dit_config():
    config = ml_collections.ConfigDict()
    config.model_name = "DiT"

    config.emb_dim = 512
    config.depth = 8
    config.num_heads = 16
    config.mlp_ratio = 2
    config.out_dim = 512

    return config



@_register
def get_dpot_config():
    config = ml_collections.ConfigDict()
    config.model_name = "DPOT"
    config.img_size = (256, 256)
    config.patch_size = 16
    config.mlp_ratio = 2.0
    config.mixing_type = 'afno'
    config.n_blocks = 4
    config.depth = 8
    config.embed_dim = 512
    config.out_dim = 1
    return config



@_register
def get_avit_config():
    config = ml_collections.ConfigDict()
    config.model_name = "AViT"
    config.out_dim = 1
    config.n_spatial_dims = 2
    config.num_groups = 8
    config.spatial_resolution = (256, 256)
    config.hidden_dim = 512
    config.num_heads = 8
    config.processor_blocks = 8
    config.drop_path = 0.0
    return config

@_register
def get_convnext_config():
    config = ml_collections.ConfigDict()
    config.model_name = "ConvNext"
    config.out_dim = 1
    config.stages = 4
    config.blocks_per_stage = 2
    config.blocks_at_neck = 2
    config.n_spatial_dims = 2
    config.init_features = 32
    return config
