import ml_collections


MODEL_CONFIGS = {}


def _register(get_config):
    """Adds reference to model config into MODEL_CONFIGS."""
    config = get_config().lock()
    name = config.get("model_name")
    MODEL_CONFIGS[name] = config
    return get_config




@_register
def get_ddpm_config():
    config = ml_collections.ConfigDict()
    config.model_name = "DDPM"
    config.out_dim = 2
    config.emb_features = 64 * 4
    config.feature_depths = (32, 64, 128, 256)
    config.attention_configs = (None, None,  {"heads": 8}, {"heads": 8})
    config.num_res_blocks = 2
    config.num_enc_blocks = 4
    config.num_middle_res_blocks = 1
    config.norm_groups = 8

    return config
