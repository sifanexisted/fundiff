# FunDiff: Diffusion Models over Function Spaces for Physics-Informed Generative Modeling

Sifan Wang, Zehao Dou, Tongrui Liu, Lu Lu

Yale University<br>

![FunDiff Pipeline](figures/pipline.png)

## Installation
To install the FunDiff package, first clone the repository and then run 

```pip install -e .```



## Data Generation

All training datasets can be downloaded from [here]() 


## Train FunDiff

Training of FunDiff consists of two steps: traininig funtion autoencoder and latent diffusion model. 
For all examples, it follows the same training procedure. For example, you can go to the kolmorogv flow genreation folder
```bash
cd examples/kf_generation
```

### Train Function Autoencoder 
```angular2html
python train_autoencoder.py --config configs/autoencoder.py:fae
```

### Train Latent Diffusion Model
```angular2html
python train_diffusion.py --config configs/diffusion.py:fae,dit
```


## Test FunDiff
To test the trained FunDiff model, you can run the following command in the same directory:

```angular2html
python eval.py --config configs/diffusion.py:fae,dit
```

