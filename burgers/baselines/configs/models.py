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
    config.out_dim = 1
    config.layer_norm_eps = 1e-6
    return config


@_register
def get_fno_config():
    config = ml_collections.ConfigDict()
    config.model_name = "FNO"
    config.emb_dim = 32
    config.modes1 = 32
    config.modes2 = 32
    config.out_dim = 1
    config.depth = 4
    return config


@_register
def get_unet_config():
    config = ml_collections.ConfigDict()
    config.model_name = "UNet"
    config.emb_dim = 32
    config.out_dim = 1
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





