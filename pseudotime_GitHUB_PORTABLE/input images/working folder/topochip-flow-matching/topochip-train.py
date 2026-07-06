# Inspired from https://github.com/w86763777/pytorch-ddpm/tree/master.

# Authors: Kilian Fatras
#          Alexander Tong

import copy
import os
import matplotlib.pyplot as plt
import torch
from absl import app, flags
from torchdyn.core import NeuralODE
from torchvision import datasets, transforms
from tqdm import trange
from utils import ema, generate_samples_topo, infiniteloop
import utils

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
from torchcfm.models.unet.unet import UNetModelWrapper
from topodataset import TopoDataSet
from unet import UNetModel_imcond_wrapper, ImCondModel

FLAGS = flags.FLAGS

flags.DEFINE_string("model", "otcfm", help="flow matching model type")
flags.DEFINE_string("output_dir", "./results/", help="output_directory")
# UNet
flags.DEFINE_integer("num_channel", 32, help="base channel of UNet")

# Training
flags.DEFINE_float("lr", 2e-4, help="target learning rate")  # TRY 2e-4
flags.DEFINE_float("grad_clip", 1.0, help="gradient norm clipping")
flags.DEFINE_integer(
    "total_steps", 400001, help="total training steps"
)  # Lipman et al uses 400k but double batch size
flags.DEFINE_integer("warmup", 5000, help="learning rate warmup")
flags.DEFINE_integer("batch_size", 16, help="batch size")  # Lipman et al uses 128
flags.DEFINE_integer("num_workers", 4, help="workers of Dataloader")
flags.DEFINE_float("ema_decay", 0.9999, help="ema decay rate")
flags.DEFINE_bool("parallel", False, help="multi gpu training")

# Evaluation
flags.DEFINE_integer(
    "save_step",
    20000,
    help="frequency of saving checkpoints, 0 to disable during training",
)
flags.DEFINE_float(
    "downsampling_factor",
    1.,
    help="factor for spatial downsampling of the images",
)
flags.DEFINE_integer(
    "limit_num_data",
    None,
    help="limit the training set size to this amount",
)

flags.DEFINE_integer(
    "crop_size",
    default=None,
    help="croping size for training images -- measured from the center",
)
flags.DEFINE_integer(
    "phalloidin_mask",
    default=0,
    help="whether to use the phalloidin mask to mask out background stain luminance",
)

flags.DEFINE_bool("imcond_inject", default=False,
                  help='whether to inject the image conditioning at each layer of the unet (True) or simply concatenate to the input (False)')

flags.DEFINE_bool("use_cfg", default=False, help='whether to use classifier free guidance for the conditioning')
flags.DEFINE_float("cfg_p_uncond", default=0.1, help='probability of unconditional training step for classifier free guidance')
flags.DEFINE_float("cfg_strength", default=1., help='classifier free guidance strength omega')
flags.DEFINE_bool('use_ddc', default=False, help='whether to use data-dependent coupling conditioning to facilitate partial observations')
flags.DEFINE_bool('threshold_input', default=False, help='whether to threshold the input images, and only keep the activations in the mask')
flags.DEFINE_float('sr', default=0, help='super resolution factor (set 0 for standard model training and not super-resolution model training)')

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup


def train(argv):
    print(
        "lr, total_steps, ema decay, save_step:",
        FLAGS.lr,
        FLAGS.total_steps,
        FLAGS.ema_decay,
        FLAGS.save_step,
    )

    print('CFG params: use_cfg, cfg_p_uncond, cfg_strength:', FLAGS.use_cfg, FLAGS.cfg_p_uncond, FLAGS.cfg_strength)
    print('using ddc:', FLAGS.use_ddc)
    print('thresholding input:', FLAGS.threshold_input)
    print('Super-resolution training:', FLAGS.sr)

    # DATASETS/DATALOADER
    dataset = TopoDataSet(
        dir='data/',
        downsampling_factor=FLAGS.downsampling_factor,
        limit_num_data=FLAGS.limit_num_data,
        crop_size=FLAGS.crop_size,
        threshold_input=FLAGS.threshold_input,
        ignore_last_10_p=True  # we train ignoring the last 10% of the data for validation purposes
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        shuffle=True,
        num_workers=FLAGS.num_workers,
        drop_last=True,
    )

    datalooper = infiniteloop(dataloader)
    x1, bg = next(datalooper)
    h, w = x1.shape[-2], x1.shape[-1]

    # MODELS
    if not FLAGS.imcond_inject:
        raise NotImplementedError('not all functionalities have been carried over to this -- probably better to adapt the other class to also cover this case...')
        net_model = ImCondModel(
            dim=(3, h, w),  # note: in local test run we had this worng (dim was at (..., 32, 32). which effectively means we were applying attention after only one time downsampling (att. resolutions was 16)
            num_res_blocks=2,
            num_channels=FLAGS.num_channel,
            channel_mult=[1, 1, 2, 2, 2],
            num_heads=4,
            num_head_channels=-1,
            attention_ds=(2**5,),
            dropout=0.1,
            use_cfg=FLAGS.use_cfg
        ).to(
            device
        )  # new dropout + bs of 128
    else:
        num_im_cond_channels = 1
        if FLAGS.use_cfg:
            num_im_cond_channels  += 1
        if FLAGS.use_ddc:
            num_im_cond_channels += 3  # 3 as the cell staining images have 3 channels
        net_model = UNetModel_imcond_wrapper(
            dim=(3, h, w),
            num_res_blocks=2,
            num_channels=FLAGS.num_channel,
            channel_mult=[1, 1, 2, 2, 2],
            num_heads=4,
            num_head_channels=-1,
            attention_ds=(2**5,),
            dropout=0.1,
            num_cond_im_channels=num_im_cond_channels,
        ).to(device)



    #################################
    #            OT-CFM
    #################################

    sigma = 0.0
    if FLAGS.model == "otcfm":
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "icfm":
        FM = ConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "fm":
        FM = TargetConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "si":
        FM = VariancePreservingConditionalFlowMatcher(sigma=sigma)
    else:
        raise NotImplementedError(
            f"Unknown model {FLAGS.model}, must be one of ['otcfm', 'icfm', 'fm', 'si']"
        )

    savedir = FLAGS.output_dir + FLAGS.model + '_' + str(FLAGS.crop_size) + '_' + str(FLAGS.downsampling_factor)+ "/"
    os.makedirs(savedir, exist_ok=True)

    # model dry run:
    with torch.no_grad():
        x0 = torch.randn_like(x1)
        if FLAGS.model == 'otcfm':
            t, xt, ut, _, bg_new = FM.guided_sample_location_and_conditional_flow(x0.to(device), x1.to(device),
                                                                                  None, bg.to(device),
                                                                                  )
        elif FLAGS.model == 'icfm':
            t, xt, ut = FM.sample_location_and_conditional_flow(x0.to(device), x1.to(device))
            bg_new = bg.to(device)
        else:
            raise NotImplementedError()
        bg_new = utils.prepare_cfg_conditioning(bg_new, prob_no_guidance=FLAGS.cfg_p_uncond) if FLAGS.use_cfg else bg_new
        dummy_mask = torch.zeros_like(xt) if FLAGS.use_ddc else None
        _ = net_model(t, xt, im_cond=bg_new, obs_mask=dummy_mask)

    #optimizers

    ema_model = copy.deepcopy(net_model)
    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    if FLAGS.parallel:
        print(
            "Warning: parallel training is performing slightly worse than single GPU training due to statistics computation in dataparallel. We recommend to train over a single GPU, which requires around 8 Gb of GPU memory."
        )
        net_model = torch.nn.DataParallel(net_model)
        ema_model = torch.nn.DataParallel(ema_model)

    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print("Model params: %.2f M" % (model_size / 1024 / 1024))

    with trange(FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            optim.zero_grad()
            x1, bg = next(datalooper)  # image of cell (three channels) and background (1 channel)
            x1, bg = x1.to(device), bg.to(device)
            if FLAGS.use_cfg:
                bg = utils.prepare_cfg_conditioning(bg, prob_no_guidance=FLAGS.cfg_p_uncond)
            x0, mask_observed = utils.get_prior_sample(x1, FLAGS.use_ddc, FLAGS.sr)

            # we permute the minibatches accordng to the OT plan, and get the noisy sample and conditional flow:
            if FLAGS.model == 'otcfm':
                t, xt, ut, _, bg_new = FM.guided_sample_location_and_conditional_flow(
                    x0, x1, None, bg
                )
                assert bg_new.shape == bg.shape
            elif FLAGS.model == 'icfm':
                t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
                bg_new = bg
            else:
                raise NotImplementedError()

            if FLAGS.use_ddc:
                assert torch.allclose(
                    ut[mask_observed.bool()], torch.Tensor([0.]).to(ut)
                ), 'this does not work, most likely you did not use icfm for ddc?'

            vt = net_model(t, xt, im_cond=bg_new,
                           obs_mask=mask_observed if FLAGS.use_ddc else None
                           )  # note: conditioned on bg_new which follows the same permutation as x1 -> xt

            loss = torch.mean((vt - ut) ** 2)
            loss.backward()
            pbar.set_description_str(f"Loss: {loss.item():.4f}")
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)  # new
            optim.step()
            sched.step()
            ema(net_model, ema_model, FLAGS.ema_decay)  # new

            # sample and Saving the weights
            if FLAGS.save_step > 0 and step % FLAGS.save_step == 0:
                #TODO: integrate masking in the ODE sampling
                # get 8x8 random conditioning images:
                im_conds = []
                x1s = []
                num = 0
                rows = 4
                cols = 4
                while num < rows*cols:
                    x1, bg = next(datalooper)
                    bg = bg.to(device)
                    im_conds.append(bg)
                    x1s.append(x1.to(device))
                    num += bg.shape[0]

                im_conds = torch.concatenate(im_conds, 0)[:rows * cols]
                x1s = torch.concatenate(x1s, 0)[:rows * cols]
                if FLAGS.use_cfg:
                    im_conds = utils.prepare_cfg_conditioning(im_conds, prob_no_guidance=0.)  # no prob of removing guidance here!

                im_conds = im_conds.reshape(rows, cols, 1 if not FLAGS.use_cfg else 2,
                                            im_conds[0].shape[-2], im_conds[0].shape[-1])
                x1s = x1s.reshape(rows, cols, 3, x1s.shape[-2], x1s.shape[-1])

                mask = torch.zeros(   # unconditional on any image data
                    (rows, cols, 3, im_conds[0].shape[-2], im_conds[0].shape[-1])
                ).to(device) if FLAGS.use_ddc else None

                generate_samples_topo(net_model, FLAGS.parallel, savedir, step, net_="normal", im_conds=im_conds,
                                      use_cfg=FLAGS.use_cfg, cfg_strength=FLAGS.cfg_strength, obs_mask=mask, x1=x1s,
                                      super_resolution_factor=FLAGS.sr)
                generate_samples_topo(ema_model, FLAGS.parallel, savedir, step, net_="ema", im_conds=im_conds,
                                      use_cfg=FLAGS.use_cfg, cfg_strength=FLAGS.cfg_strength, obs_mask=mask, x1=x1s,
                                      super_resolution_factor=FLAGS.sr)
                torch.save(
                    {
                        "net_model": net_model.state_dict(),
                        "ema_model": ema_model.state_dict(),
                        "sched": sched.state_dict(),
                        "optim": optim.state_dict(),
                        "step": step,
                    },
                    savedir + f"{FLAGS.model}_topo_weights_step_{step}.pt",
                )


if __name__ == "__main__":
    app.run(train)