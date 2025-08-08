"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import sys
sys.path.append('../')
import argparse
import numpy as np
import torch as th
import torch.distributed as dist
from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.nn import reparameterize
from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from torchvision.utils import save_image
from improved_diffusion import metrics as mt
from torchmetrics.image.fid import FrechetInceptionDistance
from improved_diffusion.nn import GaussianConvEncoderClf
# added for logging
import time

fid = FrechetInceptionDistance(feature=64)

def main():
    args = create_argparser().parse_args()
    load_classifier = args.load_classifier
    eval_disentanglement = args.eval_disentanglement
    generate_interventions = args.generate_interventions
    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    print(dist_util.dev())
    model.to(dist_util.dev())
    model.eval() # EVALUATION MODE
    
    # LOAD DATASET
    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond
        # split="test"
    )

    test_data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        split="test"
    )
    

    logger.log("test...")
    logger.log("sampling...")

    all_images = []
    all_images_thickness = []
    all_images_intensity = []
    thickness_distances = []
    intensity_distances = []
    w = None

    if load_classifier:
        logger.log("entering load_classifier...")
        # clf = GaussianConvEncoderClf(in_channels=4, latent_dim=512, num_vars=4)
        # clf.load_state_dict(th.load('../results/pendulum/classifier/classifier_angle_best.pth'))
        # clf.eval()
        if "morphomnist" in args.data_dir:
            clf1 = GaussianConvEncoderClf(in_channels=1, latent_dim=512, num_vars=2)
            clf1.load_state_dict(th.load('../results/morphomnist/classifier/classifier_thickness_best.pth'))
            clf1.eval()

            clf2 = GaussianConvEncoderClf(in_channels=1, latent_dim=512, num_vars=2)
            clf2.load_state_dict(th.load('../results/morphomnist/classifier/classifier_intensity_best.pth'))
            clf2.eval()

    if eval_disentanglement:
        logger.log("entering eval_disentanglement...")
        if "morphomnist" in args.data_dir:
            rep_train = np.empty((60000, 512))
            y_train = np.empty((60000, 2))

            train_start_time = time.time()
            batch_idx = 0
            while batch_idx < 3750:
                batch, cond = next(data)
                A = th.tensor([[0, 1], [0, 0]], dtype=th.float32).to(batch.device)

                mu, var = model.rep_emb.encode(batch.to(dist_util.dev()))
                var = th.ones(mu.shape).to(mu.device) * 0.001
                z_pre = model.causal_mask.causal_masking(mu, A)
                z_post = model.causal_mask.nonlinearity_add_back_noise(mu, z_pre).to(mu.device)

                z = reparameterize(z_post, var)
                z = z.reshape(-1, 512)
                rep_train[batch_idx*z.shape[0]:(batch_idx*z.shape[0])+z.shape[0], :] = z.cpu().detach().numpy()
                y_train[batch_idx*z.shape[0]:(batch_idx*z.shape[0])+ cond["c"].shape[0], :] = cond["c"].cpu().detach().numpy()

                if batch_idx % 50 == 0:
                    logger.log("batch_idx: " + str(batch_idx) + "/3750")
                batch_idx += 1

            logger.log(f'{time.strftime("%H:%M:%S" , time.localtime())} -> Training time: {time.time() - train_start_time:.2f}')
            test_start_time = time.time()

            rep_test = np.empty((10000, 512))
            y_test = np.empty((10000, 2))
            batch_idx = 0
            while batch_idx < 625:
                batch, cond = next(test_data)
                A = th.tensor([[0, 1], [0, 0]], dtype=th.float32).to(batch.device)

                mu, var = model.rep_emb.encode(batch.to(dist_util.dev()))
                var = th.ones(mu.shape).to(mu.device) * 0.001
                z_pre = model.causal_mask.causal_masking(mu, A)
                z_post = model.causal_mask.nonlinearity_add_back_noise(mu, z_pre).to(mu.device)

                z = reparameterize(z_post, var)

                z = z.reshape(-1, 512)
                rep_test[batch_idx*z.shape[0]:(batch_idx*z.shape[0])+z.shape[0], :] = z.cpu().detach().numpy()
                y_test[batch_idx*z.shape[0]:(batch_idx*z.shape[0])+ cond["c"].shape[0], :] = cond["c"].cpu().detach().numpy()

                if batch_idx % 50 == 0:
                    logger.log("batch_idx: " + str(batch_idx) + "/600")
                batch_idx += 1

            logger.log(f'{time.strftime("%H:%M:%S" , time.localtime())} -> Testing time: {time.time() - test_start_time:.2f}')
            logger.log(f'{time.strftime("%H:%M:%S" , time.localtime())} -> Computing DCI metrics (this will take far longer than the previous 2 steps combined)...')
            dci_start_time = time.time()
            scores, importance_matrix, code_importance = mt._compute_dci(rep_train.T, y_train.T, rep_test.T, y_test.T)
            logger.log(f'{time.strftime("%H:%M:%S" , time.localtime())} -> DCI time: {time.time() - dci_start_time:.2f}')
            logger.log(f'{time.strftime("%H:%M:%S" , time.localtime())} -> DCI scores: {scores}')
            logger.log(f'{time.strftime("%H:%M:%S" , time.localtime())} -> Total time: {time.time() - train_start_time:.2f}')
    else:
        logger.log("NOT entering eval_disentanglement...")
        counter = 0
        while len(all_images) * args.batch_size < args.num_samples:
            batch, cond = next(data)

            # mu, var = model.rep_emb.encode(batch.to(dist_util.dev()))
            if "morphomnist" in args.data_dir:
                A = th.tensor([[0, 1], [0, 0]], dtype=th.float32).to(batch.device)

                # THICKNESS INTERVENTIONS
                mu, var = model.rep_emb.encode(batch.to(dist_util.dev()))
                var = th.ones(mu.shape).to(mu.device) * 0.001
                
                mu[:, :256] = th.ones((args.batch_size, 256)) * 0.2

                z_pre = model.causal_mask.causal_masking(mu, A)
                z_post = model.causal_mask.nonlinearity_add_back_noise(mu, z_pre).to(mu.device)

                z = reparameterize(z_post, var)

                noise = th.randn_like(batch).to(dist_util.dev())
                t = th.ones((batch.shape[0]), dtype=th.int64) * 249
                t = t.to(dist_util.dev())

                x_t = diffusion.q_sample(batch.to(dist_util.dev()), t, noise=noise)
                sample_fn = (
                    diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
                )
                cond["z"] = z
                cond["y"] = cond["y"].to(dist_util.dev())

                t = time.time()
                logger.log(f"{time.strftime("%H:%M:%S", time.localtime())} -> Before sample_fn")
                sample = sample_fn(
                    model,
                    (args.batch_size, 1, args.image_size, args.image_size),
                    noise=x_t,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=cond,
                    w=w
                )
                logger.log(f'{time.strftime("%H:%M:%S", time.localtime())} -> After sample_fn | Total time for 1 step: {time.time() - t:.2f}')

                gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
                all_images_thickness.extend([sample.cpu().numpy() for sample in gathered_samples])

                logger.log("Before intensity intervention...")

                # INTENSITY INTERVENTIONS
                mu, var = model.rep_emb.encode(batch.to(dist_util.dev()))
                var = th.ones(mu.shape).to(mu.device) * 0.001

                z_pre = model.causal_mask.causal_masking(mu, A)
                z_post = model.causal_mask.nonlinearity_add_back_noise(mu, z_pre).to(mu.device)
                z_post[:, 256:] = th.ones((args.batch_size, 256)) * 0.2
                z = reparameterize(z_post, var)

                x_t = diffusion.q_sample(batch.to(dist_util.dev()), t, noise=noise)

                sample_fn = (
                    diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
                )
                cond["z"] = z
                cond["y"] = cond["y"].to(dist_util.dev())

                sample = sample_fn(
                    model,
                    (args.batch_size, 1, args.image_size, args.image_size),
                    noise=x_t,
                    clip_denoised=args.clip_denoised,
                    model_kwargs=cond,
                    w=w
                )

                gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
                all_images_intensity.extend([sample.cpu().numpy() for sample in gathered_samples])
                print(f'Batch {counter}/{args.num_samples} complete!')
            break

        if generate_interventions:
            logger.log("generating interventions...")
            if "morphomnist" in args.data_dir:
                save_image(batch[:32], "../results/morphomnist/original.png")
                logger.log("batch saved as ../results/morphomnist/original.png")

                # SAVE THICKNESS INTERVENED IMAGE
                arr = np.concatenate(all_images_thickness, axis=0)
                arr = arr[: args.num_samples]
                temp = arr[:32]

                temp = th.tensor(temp, dtype=th.float32)
                # save_image(temp, '../results/morphomnist/causaldiffae_masked/intervene_thickness.png')
                save_image(temp, f'../results/morphomnist/intervene_thickness_w={w}.png')
                logger.log(f'batch saved as ../results/morphomnist/intervene_thickness_w={w}.png')

                # SAVE INTENSITY INTERVENED IMAGE
                arr = np.concatenate(all_images_intensity, axis=0)
                arr = arr[: args.num_samples]
                temp = arr[:32]

                temp = th.tensor(temp, dtype=th.float32)
                # save_image(temp, '../results/morphomnist/causaldiffae_masked/intervene_intensity.png')
                save_image(temp, f'../results/morphomnist/intervene_intensity_w={w}.png')
                logger.log(f'batch saved as ../results/morphomnist/intervene_intensity_w={w}.png')
        else:
            if "morphomnist" in args.data_dir:
                mean_dist = th.tensor(sum(thickness_distances) / len(thickness_distances))
                gathered_samples = [th.zeros_like(mean_dist) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_samples, mean_dist)  # gather not supported with NCCL
                print(f"Thickness MAE: {sum(gathered_samples) / len(gathered_samples)}")
                
                mean_dist = th.tensor(sum(intensity_distances) / len(intensity_distances))
                gathered_samples = [th.zeros_like(mean_dist) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_samples, mean_dist)  # gather not supported with NCCL
                print(f"Intensity MAE: {sum(gathered_samples) / len(gathered_samples)}")
    
    dist.barrier()
    logger.log("testing complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        data_dir="",
        rep_cond=False,
        in_channels=3,
        n_vars=2,
        load_classifier=False,  # added arguments after this
        eval_disentanglement=True,
        generate_interventions=True,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()