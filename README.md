# Causal Diffusion Autoencoders: Toward Counterfactual Generation via Diffusion Probabilistic Models

This fork contains the source code for the implementation of "Causal Diffusion Autoencoders: Toward Counterfactual Generation via Diffusion Probabilistic Models" (ECAI 2024). The original code lacked a clear README file, making it unclear how to obtain the desired dataset and where to find the appropriate files. This fork aims to address those issues. This code improves upon the version of the original source code that was the latest version in Summer 2025, namely commit `4e9f669` (`4e9f669ef75ac2d86eb31f9deb08423a159449c4`).

This README describes the process of transforming [the original source code](https://github.com/Akomand/CausalDiffAE) into this repository and provides a step-by-step guide for running the code on the MorphoMNIST dataset. Note that if you clone this repository, every file that needed to be changed has already been changed. Except for dataset generation, everything is ready. Even if you decide to simply clone this fork and avoid all the trouble of converting the original source code into working (for MorphoMNIST) code, you will still have to refer to the Synthetic MorphoMNIST Data Generation section.

# Getting environment up and running
1. Create a virtual environment: `python3.12 -m venv causaldiffae_venv`. If virtual environment is not activated, activate it with `.\causaldiffae_venv\Scripts\activate` on Windows and `source causaldiffae_venv/bin/activate` for Linux.


2. Install the requirements with this huge `pip install` command: 
```
pip install absl-py==2.3.1 aiohappyeyeballs==2.6.1 aiohttp==3.12.14 aiosignal==1.4.0 attrs==25.3.0 blobfile==3.0.0 colorama==0.4.6 contourpy==1.3.2 cycler==0.12.1 filelock==3.18.0 fonttools==4.59.0 frozenlist==1.7.0 fsspec==2025.5.1 grpcio==1.73.1 idna==3.10 imageio==2.37.0 Jinja2==3.1.6 joblib==1.5.1 kiwisolver==1.4.8 lazy_loader==0.4 lightning-utilities==0.14.3 lxml==6.0.0 Markdown==3.8.2 MarkupSafe==3.0.2 matplotlib==3.10.3 mpi4py==4.1.0 mpmath==1.3.0 multidict==6.6.3 networkx==3.5 numpy==2.3.1 opt_einsum==3.4.0 packaging==25.0 pandas==2.3.1 pillow==11.3.0 propcache==0.3.2 protobuf==6.31.1 pycryptodomex==3.23.0 pyparsing==3.2.3 pyro-api==0.1.2 pyro-ppl==1.9.1 python-dateutil==2.9.0.post0 pytorch-lightning==2.5.2 pytz==2025.2 PyYAML==6.0.2 scikit-image==0.25.2 scikit-learn==1.7.1 scipy==1.16.0 seaborn==0.13.2 setuptools==80.9.0 six==1.17.0 sympy==1.14.0 tensorboard==2.20.0 tensorboard-data-server==0.7.2 threadpoolctl==3.6.0 tifffile==2025.6.11 torch==2.7.1 torch-fidelity==0.3.0 torchmetrics==1.8.0 torchvision==0.22.1 tqdm==4.67.1 typing_extensions==4.14.1 tzdata==2025.2 urllib3==2.5.0 Werkzeug==3.1.3 yarl==1.20.1
```

3. Install MPI for your operating system. Example for Debian-based Linux distributions: `sudo apt install openmpi-bin`

# Synthetic MorphoMNIST Data Generation

This model uses the MorphoMNIST dataset with the intensity attribute, which is not included in any of the datasets from the [Morpho-MNIST repository](https://github.com/dccastro/Morpho-MNIST). Therefore, we provide instructions for generating the appropriate dataset.

1. Download MorphoMNIST Dataset. Go to the [Morpho-MNIST repository](https://github.com/dccastro/Morpho-MNIST) and download any dataset (the original work used `plain`). After extraction, confirm the folder structure is correct and not nested (should look like `plain/FILES`, not `plain/plain/FILES`).


2. Clone [DeepSCM repository](https://github.com/biomedia-mira/deepscm?tab=readme-ov-file).


3. Copy the `Morpho-MNIST/morphomnist` subfolder into the `deepscm` directory so it is located at `deepscm/morphomnist`. If there is a collision (as there is a file `morphomnist` in `deepscm/`), simply overwrite the existing file with this folder.

If you plan to use DeepSCM code for other purposes, sync the submodule `git submodule update --recursive --init`

4. Open `deepscm/datasets/morphomnist/transforms.py` (`deepscm/morphomnist` and `deepscm/datasets/morphomnist` are separate folders, great naming!) and on line 11, remove the `multichannel=False` argument.  
Change: ```disk = transform.pyramid_reduce(mag_disk, downscale=scale, order=1, multichannel=False)```
to: ```disk = transform.pyramid_reduce(mag_disk, downscale=scale, order=1)```


5. To generate a synthetic MorphoMNIST dataset with the intensity attribute, run the following command, replacing the paths with those appropriate to your environment:
```
python -m deepscm.datasets.morphomnist.create_synth_thickness_intensity_data --data-dir /path/to/morphomnist -o /path/to/dataset
```

***After getting the synthetic MorphoMNIST dataset, you can remove the entire `deepscm` from your computer.***

# Training

1. Clone the repo: 
```
git clone https://github.com/Akomand/CausalDiffAE.git
cd CausalDiffAE
```

If you want to clone this fork with all the files set up, simply: ```git clone https://github.com/Stochastic-Batman/CausalDiffAE_FZJ.git```.

If you haven't installed dependencies with pip by following this README.md file, you can create environment with conda:
```
conda env create -f environment.yml
```

2. The authors have the synthetic MorphoMNIST dataset in the `morphomnist` directory, under the `datasets` folder:
```
causaldiffae_venv/
datasets/
	└── morphomnist/
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

Copy [`io.py` from Morpho-MNIST repo](https://github.com/dccastro/Morpho-MNIST/blob/main/morphomnist/io.py) under `datasets/` folder and in `improved_diffusion/image_datasets.py` change:
```
# from datasets.morphomnist import io
import io
```
to:
```
from datasets import io
# import io
```
To import this `io.py`, please add empty `datasets/__init__.py`.

3. Specify Causal Adjacency Matrix A in `improved_diffusion/unet.py`
```
A = th.tensor([[0, 1], [0, 0]], dtype=th.float32)
```

4. For each of the training and testing scripts in `scripts\morhomnist`(unfortunate typo in the original code) and other(`scripts\` subfolders) set `--data-dir` argument to `../datasets/morphomnist`.


5. In `improved_diffusion/nn.py`, find the `GaussianConvEncoder` class. In the ``__init__`` method, change:
```
self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)
```
to:
```
hidden_dims_last = hidden_dims[-1]
self.fc_mu = nn.Linear(hidden_dims_last, latent_dim)
self.fc_var = nn.Linear(hidden_dims_last, latent_dim)
```

6. Optionally, add `steps=float("inf")` argument to the `TrainLoop` class in `improved_diffusion/train_util.py` before `use_fp16=False,` and modify the `run_loop` method in the following way:
```
def run_loop(self):
    # THIS IS THE TOTAL NUMBER OF ITERATIONS
    logger.log("entering the training loop...")
    while (
        (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps)
        and self.step < self.steps
    ):
        batch, cond = next(self.data)
        
        self.run_step(batch, cond) # RUN FIRST STEP WITH BATCH AND CONDITION (IF ANY)
        if self.step % self.log_interval == 0:
            logger.dumpkvs()
        if self.step % self.save_interval == 0:
            self.save()
            # Run for a finite amount of time in integration tests.
            if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                return
        
        self.step += 1
        # KL WEIGHT SCHEDULER
        weight = self.linear_kl_weight_scheduler(self.step, 5000, 0.0, 1.0)  # any values here you would like
        self.diffusion.kl_weight = weight
        logger.log(f"step {self.step} complete!")
        
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()
```
You will also need to add `steps=float("inf"),` in `scripts/image_train.py` to `defaults` dictionary of `create_argparser()` before `use_fp16=False`.

7. If you want to run on a single machine (with `mpiexec -n 1`) and you want to save the models, for the `save` method of the `TrainLoop` class in `improved_diffusion/train_util.py` change `if dist.get_rank() == 1:` to `if dist.get_rank() == 0:`.


8. In the `main` method of `scripts/image_train.py`, comment out the loggers for the other examples:

```
dist_util.setup_dist()
logger.configure(dir = "../results/morphomnist")
logger.configure(dir = "../results/morphomnist/causaldiffae_masked_p")
logger.configure(dir = "../results/morphomnist/diffae_unaligned")

# logger.configure(dir = "../results/pendulum/causaldiffae_masked")
# logger.configure(dir = "../results/pendulum/label_conditional")
# logger.configure(dir = "../results/pendulum/diffae_aligned")
# logger.configure(dir = "../results/pendulum/causaldiffae")
# logger.configure(dir = "../results/pendulum/diffae_unaligned")

# logger.configure(dir = "../results/circuit/causaldiffae_masked")
# logger.configure(dir = "../results/circuit/diffae_unaligned")
# logger.configure(dir = "../results/circuit/diffae")
# logger.configure(dir = "../results/circuit/label_conditional")
```

9. In `improved_diffusion/dist_util.py`, uncomment the line choosing the backend:  
``backend = "gloo" if not th.cuda.is_available() else "nccl"``. **In case you get an error related to CUDA memory, set `backend = "gloo"`.**


10. Navigate to the `scripts` folder, specify hyperparameters or run the default training script:
```
./[dataset]/train_[dataset]_causaldae.sh
```
or directly run (remove `--steps 100` if you did not add that parameter):
```
mpiexec -n 1 python image_train.py --data_dir ../datasets/morphomnist --n_vars 2 --in_channels 1 --image_size 28 --num_channels 128 --causal_modeling True --num_res_blocks 3 --learn_sigma False --class_cond True --rep_cond True --flow_based False --masking True --diffusion_steps 20 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 1e-4 --batch_size 128 --steps 100
```

Example (the dashed lines are reversed because the script is run from Windows PowerShell):
```
.\morhomnist\train_mnist_causaldae.sh
```
11. For classifier-free paradigm training, set `masking=True` in hyperparameter configs.

**Note:** this statement is from README of the original repository. However, it turns out, `masking` is not used anywhere except `self.masking = masking`. Basically, this line has no effect on the code.


12. To train anti-causal classifiers to evaluate effectiveness, navigate to `improved_diffusion` (or whichever example you are experimenting with) and run:
```
python [dataset]_classifier.py
```

you might need to import and use different dataloaders (by default, `get_dataloader_pendulum` was loaded, I had to load `get_dataloader_morphomnist`).

# Testing & Counterfactual Generation

1. In `scripts/image_causaldae_test.py` comment out all the import that start with `from datasets.generators import ...`.


2. In `improved_diffusion/metrics.py`, change `from munkres import Munkres` to `from .munkres import Munkres` (notice dot in front `munkres` for relative import). 


3. In every test script under `scripts/morhomnist`, change the path `../results/morphomnist/causaldiffae` to `../results/morphomnist`.


4. Comment out the entire body of 
```
elif "pendulum" in args.data_dir: 
    A = th.tensor([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=th.float32).to(batch.device) 
```
and add ``pass`` instead of it.

5. There might be several lines with `z = reparameterize(z_post, var)`, but `z_post` is commented on top of them. Uncomment them.


6. From `scripts/image_causaldae_test.py` remove `load_classifier = True`. Add `load_classifier=False` to the default dict of `create_argparser()` in the same file. In the `main` method, add `load_classifier` after `args = create_argparser().parse_args()`.


7. Do the same for `eval_disentanglement = False`, `generate_interventions = True` and `w=1`.


8. At this point, it would probably be a good idea to simply copy `scripts/image_causaldae_test.py` from this fork.


9. For counterfactual generation, run the following script with the specified causal graph:
```
./morhomnist/test_[dataset]_causaldae.sh
```
or
```
python image_causaldae_test.py --data_dir ../datasets/morphomnist --model_path ../results/morphomnist/model002000.pt --n_vars 2 --in_channels 1 --image_size 28 --num_channels 128 --num_res_blocks 3 --learn_sigma False --class_cond True --causal_modeling True --rep_cond True --diffusion_steps 1000 --batch_size 16 --timestep_respacing 250 --use_ddim True
```

10. Modify `image_causaldae_test.py` to perform desired intervention and sample counterfactual.

**Note:** The code in this repository contains additional logging statements, which are only for debugging purposes and have nothing to do with the functional requirements of the code. For convenience, I also removed everything extra from `image_causaldae_test.py` and only left parts related to the MorphoMNIST testing. However, if you follow the README up until now from the original code, you will still have code for pendulum and circuit experiments.

### Data acknowledgements
Experiments are run on the following dataset to evaluate our model:

<details closed>
<summary>MorphoMNIST Dataset</summary>

[Link to dataset](https://github.com/dccastro/Morpho-MNIST)
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
