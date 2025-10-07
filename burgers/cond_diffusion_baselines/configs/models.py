import ml_collections


MODEL_CONFIGS = {}


def _register(get_config):
    """Adds reference to model config into MODEL_CONFIGS."""
    config = get_config().lock()
    name = config.get("model_name")
    MODEL_CONFIGS[name] = config
    return get_config


@_register
def get_vedps_config():
    config = ml_collections.ConfigDict()
    config.model_name = "VEPrecond"
    config.img_resolution = 256
    config.img_channels = 1
    config.label_dim = 0
    config.use_fp16 = False
    # config.sigma_min = 8e-2
    # config.sigma_max = 200.0
    config.sigma_min = 0.002
    config.sigma_max = 80.0

    config.model_kwargs = dict(
        embedding_type="positional",
        model_channels=64,
        dropout=0.0,
    )
    return config


#
#
# @_register
# def get_vedps_config():
#     config = ml_collections.ConfigDict()
#     config.model_name = "VEPrecond"
#     config.img_channels = 1
#     config.emb_features = 64
#     config.feature_depths = (64, 128, 256, 512)
#     config.attention_configs = (None, None, None, {"heads": 8})
#     config.num_res_blocks = 1
#     config.num_middle_res_blocks = 1
#     config.norm_groups = 8
#
#     return config
