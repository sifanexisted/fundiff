import ml_collections


MODEL_CONFIGS = {}


def _register(get_config):
    """Adds reference to model config into MODEL_CONFIGS."""
    config = get_config().lock()
    name = config.get("model_name")
    MODEL_CONFIGS[name] = config
    return get_config


@_register
def get_vit_config():
    config = ml_collections.ConfigDict()
    config.model_name = "ViT"
    config.patch_size = (16, 16)
    config.emb_dim = 256
    config.depth = 8
    config.num_heads = 8
    config.mlp_ratio = 2
    config.out_dim = 2
    config.layer_norm_eps = 1e-6
    return config



@_register
def get_fno_config():
    config = ml_collections.ConfigDict()
    config.model_name = "FNO"
    config.emb_dim = 32
    config.modes1 = 32
    config.modes2 = 32
    config.out_dim = 2
    config.depth = 4
    return config



@_register
def get_unet_config():
    config = ml_collections.ConfigDict()
    config.model_name = "UNet"
    config.emb_dim = 32
    config.out_dim = 2
    return config
