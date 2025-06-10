# FunDiff: Diffusion Models over Function Spaces for Physics-Informed Generative Modeling

**Authors:** Sifan Wang*, Zehao Dou*, Tongrui Liu, Lu Lu  
*Equal contribution  

**Affiliation:** Yale University

[//]: # (![FunDiff Pipeline]&#40;figures/pipline.png&#41;)

[//]: # (<img src="figures/pipline.png" alt="FunDiff Pipeline" width="800" />)

## Installation

To install the FunDiff package, clone the repository and install in development mode:

```bash
git clone [repository-url]
cd fundiff
pip install -e .
```

## Dataset

Training datasets are available for download at [dataset link](). Download and extract the datasets to the appropriate directory and change the data path accordinly in config files before training.

## Training Pipeline

FunDiff employs a two-stage training procedure:

1. **Training Function Autoencoder**: Maps functions to a latent representation 
2. **Training Latent Diffusion Model**: Learns the diffusion process in the latent space via rectified flow

### Example: Kolmogorov Flow Generation

Navigate to the Kolmogorov flow example directory:

```bash
cd examples/kf_generation
```

#### Stage 1: Train Function Autoencoder

```bash
python train_autoencoder.py --config configs/autoencoder.py:fae
```

This stage learns to encode physical functions into a compact latent representation while preserving essential structural information.

#### Stage 2: Train Latent Diffusion Model

```bash
python train_diffusion.py --config configs/diffusion.py:fae,dit
```

This stage trains the diffusion model to generate new samples in the learned latent space.

## Evaluation

To evaluate the trained FunDiff model and generate new samples:

```bash
python eval.py --config configs/diffusion.py:fae,dit
```

This command loads the trained models and generates samples according to the specified configuration.

<!-- ## Configuration

The training and evaluation processes are controlled through configuration files located in the `configs/` directory. Key parameters include:

- **fae**: Function autoencoder configuration
- **dit**: Diffusion transformer configuration

Modify these configurations to adjust model architecture, training hyperparameters, and evaluation settings. -->

## Repository Structure

```
FunDiff/
├── function_diffusion/         # source code of function diffusion implementation
├── burgers/                    # Burgers equation examples
├── damp_sine/                  # Damped sine wave examples
├── figures/                    # pipline figure
├── kf_generation/              # Kolmogorov flow generation
├── kf_reconstruction/          # Kolmogorov flow reconstruction
├── linear_elasticity/          # Linear elasticity problems
├── turbulence_mass_transfer/   # Turbulence and mass transfer examples
├── requirements.txt            # Python dependencies
└── setup.py                    # Package installation script
```

## Citation

If you use FunDiff in your research, please cite:

```bibtex
[Citation information to be added]
```

