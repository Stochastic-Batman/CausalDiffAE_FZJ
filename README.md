# Causal Diffusion Autoencoders: Toward Counterfactual Generation via Diffusion Probabilistic Models
This is the source code for the implementation of "Causal Diffusion Autoencoders: Toward Counterfactual Generation via Diffusion Probabilistic Models" (ECAI 2024)

Diffusion probabilistic models (DPMs) have become the state-of-the-art in high-quality image generation. However, DPMs have an arbitrary noisy latent space with no interpretable or controllable semantics. Although there has been significant research effort to improve image sample quality, there is little work on representation-controlled generation using diffusion models. Specifically, causal modeling and controllable counterfactual generation using DPMs is an underexplored area. In this work, we propose CausalDiffAE, a diffusion-based causal representation learning framework to enable counterfactual generation according to a specified causal model. Our key idea is to use an encoder to extract high-level semantically meaningful causal variables from high-dimensional data and model stochastic variation using reverse diffusion. We propose a causal encoding mechanism that maps high-dimensional data to causally related latent factors and parameterize the causal mechanisms among latent factors using neural networks. To enforce the disentanglement of causal variables, we formulate a variational objective and leverage auxiliary label information in a prior to regularize the latent space. We propose a DDIM-based counterfactual generation procedure subject to do-interventions. Finally, to address the limited label supervision scenario, we also study the application of CausalDiffAE when a part of the training data is unlabeled, which also enables granular control over the strength of interventions in generating counterfactuals during inference. We empirically show that CausalDiffAE learns a disentangled latent space and is capable of generating high-quality counterfactual images.

![Model](causaldiffae.png)

# Getting environment up and running
1. Create a virtual environment: `python3.12 -m venv causaldiffae_venv`
2. Install the requirement with this huge command: 
```
pip install absl-py==2.3.1 aiohappyeyeballs==2.6.1 aiohttp==3.12.14 aiosignal==1.4.0 attrs==25.3.0 blobfile==3.0.0 colorama==0.4.6 contourpy==1.3.2 cycler==0.12.1 filelock==3.18.0 fonttools==4.59.0 frozenlist==1.7.0 fsspec==2025.5.1 grpcio==1.73.1 idna==3.10 imageio==2.37.0 Jinja2==3.1.6 joblib==1.5.1 kiwisolver==1.4.8 lazy_loader==0.4 lightning-utilities==0.14.3 lxml==6.0.0 Markdown==3.8.2 MarkupSafe==3.0.2 matplotlib==3.10.3 mpi4py==4.1.0 mpmath==1.3.0 multidict==6.6.3 networkx==3.5 numpy==2.3.1 opt_einsum==3.4.0 packaging==25.0 pandas==2.3.1 pillow==11.3.0 propcache==0.3.2 protobuf==6.31.1 pycryptodomex==3.23.0 pyparsing==3.2.3 pyro-api==0.1.2 pyro-ppl==1.9.1 python-dateutil==2.9.0.post0 pytorch-lightning==2.5.2 pytz==2025.2 PyYAML==6.0.2 scikit-image==0.25.2 scikit-learn==1.7.1 scipy==1.16.0 seaborn==0.13.2 setuptools==80.9.0 six==1.17.0 sympy==1.14.0 tensorboard==2.20.0 tensorboard-data-server==0.7.2 threadpoolctl==3.6.0 tifffile==2025.6.11 torch==2.7.1 torchmetrics==1.8.0 torchvision==0.22.1 tqdm==4.67.1 typing_extensions==4.14.1 tzdata==2025.2 urllib3==2.5.0 Werkzeug==3.1.3 yarl==1.20.1
```

# MorphoMNIST Setup

This model uses the MorphoMNIST dataset with the intensity attribute, which is not included in any of the datasets from the [Morpho-MNIST repository](https://github.com/dccastro/Morpho-MNIST). Therefore, we provide instructions for generating the appropriate dataset.

1. Download MorphoMNIST Dataset. Go to the [Morpho-MNIST repository](https://github.com/dccastro/Morpho-MNIST) and download any dataset (the original work used `plain`). After extraction, confirm the folder structure is correct and not nested (should look like `plain/FILES`, not `plain/plain/FILES`).

2. Clone [DeepSCM repository](https://github.com/biomedia-mira/deepscm?tab=readme-ov-file).

3. Copy the `Morpho-MNIST/morphomnist` subfolder into the `deepscm` directory so it is located at `deepscm/morphomnist`.

4. Install requirements (if you encounter Python version issues, use Python 3.7.2) with either `pip install -r requirements.txt` or 
```
pip install numpy pandas pyro-ppl pytorch-lightning scikit-image scikit-learn scipy seaborn tensorboard torch torchvision
```

If you plan to use DeepSCM code for other purposes, sync the submodule `git submodule update --recursive --init`


5. Open `deepscm/datasets/morphomnist/transforms.py` (`deepscm/morphomnist` and `deepscm/datasets/morphomnist` are separate folders, great naming!) and on line 11, remove the `multichannel=False` argument.  
Change: ```disk = transform.pyramid_reduce(mag_disk, downscale=scale, order=1, multichannel=False)```
to: ```disk = transform.pyramid_reduce(mag_disk, downscale=scale, order=1)``` 

6. To generate a synthetic MorphoMNIST dataset with the intensity attribute, run the following command, replacing the paths with those appropriate to your environment:
```
python -m deepscm.datasets.morphomnist.create_synth_thickness_intensity_data --data-dir /path/to/morphomnist -o /path/to/dataset
```

## After getting the synthetic MorphoMNIST dataset, you can remove the entire `deepscm` from your computer.

# Getting CausalDiffAE up and running

1. Clone the repo: 
```
git clone https://github.com/Akomand/CausalDiffAE.git
cd CausalDiffAE
```

If you haven't installed dependencies with pip by following this README.md file, you can create environment with conda:
```
conda env create -f environment.yml
```

2. The authors have the synthetic MorphoMNIST dataset at the top level of the repository, under the `datasets` folder:
```
causaldiffae_venv/
datasets/
├── args.txt
├── t10k-images-idx3-ubyte.gz
├── t10k-labels-idx1-ubyte.gz
├── t10k-morpho.csv
├── train-images-idx3-ubyte.gz
├── train-labels-idx1-ubyte.gz
└── train-morpho.csv
improved_diffusion/
scripts/
OTHER FILES
```

Copy [`io.py` from Morpho-MNIST repo](https://github.com/dccastro/Morpho-MNIST/blob/main/morphomnist/io.py) under `datasets/` folder and change:
```
# from datasets.morphomnist import io
import io
```
to:
```
from datasets.morphomnist import io
# import io
```

3. Create Dataset in `improved_diffusion/image_datasets.py`

4. Specify Causal Adjacency Matrix A in `improved_diffusion/unet.py`
```
A = th.tensor([[0, 1], [0, 0]], dtype=th.float32)
```

5. Specify hyperparameters and run training script:
```
./train_[dataset]_causaldae.sh
```

6. For classifier-free paradigm training, set `masking=True` in hyperparameter configs.

7. To train anti-causal classifiers to evaluate effectiveness, run:
```
python [dataset]_classifier.py
```

8. For counterfactual generation, run the following script with the specified causal graph:
```
./test_[dataset]_causaldae.sh
```

9. Modify `image_causaldae_test.py` to perform desired intervention and sample counterfactual.

### Data acknowledgements
Experiments are run on the following datasets to evaluate our model:

#### Datasets
<details closed>
<summary>MorphoMNIST Dataset</summary>

[Link to dataset](https://github.com/dccastro/Morpho-MNIST)
</details>

<details closed>
<summary>Pendulum Dataset</summary>

[Link to dataset](https://github.com/huawei-noah/trustworthyAI/tree/master/research/CausalVAE/causal_data)
</details>

<details closed>
<summary>CausalCircuit Dataset</summary>

[Link to dataset](https://developer.qualcomm.com/software/ai-datasets/causalcircuit)
</details>

## Citation

If you use our code or think our work is relevant to yours, we encourage you to cite this paper:

```bibtex
@inproceedings{
komanduri2024causaldiffae,
title={Causal Diffusion Autoencoders: Toward Counterfactual Generation via Diffusion Probabilistic Models},
author={Aneesh Komanduri and Chen Zhao and Feng Chen and Xintao Wu},
booktitle={Proceedings of the 27th European Conference on Artificial Intelligence},
year={2024}
}
```
