import scipy.io

from function_diffusion.utils.data_utils import BaseDataset

def create_dataset(config):
    data = scipy.io.loadmat('/scratch/sifanw/transformer_as_integrator/burgers/burger_nu_1e-3.mat')

    num_train = config.dataset.num_train_samples

    usols = data['output']
    usols = usols[:, :-1, :, None]

    u_train = usols[:num_train]
    u_test = usols[num_train:]

    train_dataset = BaseDataset(u_train, config.dataset.downsample_factor)
    test_dataset = BaseDataset(u_test, config.dataset.downsample_factor)

    return train_dataset, test_dataset



