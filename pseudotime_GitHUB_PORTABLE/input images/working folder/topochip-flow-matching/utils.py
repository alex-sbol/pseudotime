import copy
import os
import numpy as np
import torch
from torch import distributed as dist
from torchdyn.core import NeuralODE
import matplotlib.pyplot as plt
from functools import partial
from torchvision.transforms import ToPILImage, Resize
from torchvision.utils import make_grid
from typing import Callable, Union, List, BinaryIO, Optional
import pathlib
import torch.nn.functional as F
import torch.nn as nn

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def setup(
    rank: int,
    total_num_gpus: int,
    master_addr: str = "localhost",
    master_port: str = "12355",
    backend: str = "nccl",
):
    """Initialize the distributed environment.

    Args:
        rank: Rank of the current process.
        total_num_gpus: Number of GPUs used in the job.
        master_addr: IP address of the master node.
        master_port: Port number of the master node.
        backend: Backend to use.
    """

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    # initialize the process group
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=total_num_gpus,
    )


def generate_samples(model, parallel, savedir, step, net_="normal"):
    """Save 64 generated images (8 x 8) for sanity check along training.

    Parameters
    ----------
    model:
        represents the neural network that we want to generate samples from
    parallel: bool
        represents the parallel training flag. Torchdyn only runs on 1 GPU, we need to send the models from several GPUs to 1 GPU.
    savedir: str
        represents the path where we want to save the generated images
    step: int
        represents the current step of training
    """
    model.eval()

    model_ = copy.deepcopy(model)
    if parallel:
        # Send the models from GPU to CPU for inference with NeuralODE from Torchdyn
        model_ = model_.module.to(device)

    node_ = NeuralODE(model_, solver="euler", sensitivity="adjoint")
    with torch.no_grad():
        traj = node_.trajectory(
            torch.randn(64, 3, 32, 32, device=device),
            t_span=torch.linspace(0, 1, 100, device=device),
        )
        traj = traj[-1, :].view([-1, 3, 32, 32]).clip(-1, 1)
        traj = traj / 2 + 0.5
    save_image(traj, savedir + f"{net_}_generated_FM_images_step_{step}.png", nrow=8)

    model.train()

def generate_samples_topo(model, parallel, savedir, step, net_="normal", im_conds=None, num_int_steps=100,
                          return_whole_traj=False, use_cfg=False, cfg_strength=1., obs_mask=None, x1=None,
                          super_resolution_factor=0):
    """Save generated images (8 x 8) for sanity check along training. We save in the shape of im_conds

    Parameters
    ----------
    model:
        represents the neural network that we want to generate samples from
    parallel: bool
        represents the parallel training flag. Torchdyn only runs on 1 GPU, we need to send the models from several GPUs to 1 GPU.
    savedir: str
        represents the path where we want to save the generated images
    step: int
        represents the current step of training
    im_conds: torch.Tensor: (num_rows, num_cols, C, H, W)
        represents the conditioning images. The shape (num_rows, num_cols) will be the subplot shape
    """
    model.eval()
    num_rows_plot, num_cols_plot, c_cond, h, w = im_conds.shape

    model_ = copy.deepcopy(model)
    if parallel:
        # Send the models from GPU to CPU for inference with NeuralODE from Torchdyn
        model_ = model_.module.to(device)

    im_conds_flat = im_conds.view(-1, c_cond, h, w)
    obs_mask = obs_mask.view(-1, 3, h, w) if obs_mask is not None else None
    fw_fn_cond = partial(model_.forward, im_cond=im_conds_flat, obs_mask=obs_mask)
    if not use_cfg:
        fw_fn = fw_fn_cond
    else:
        im_conds_flat_uncond = prepare_cfg_conditioning(im_conds_flat[:, :1], prob_no_guidance=1.)
        fw_fn_uncond = partial(model_.forward, im_cond=im_conds_flat_uncond)
        def fw_fn(t, x, *args, **kwargs):
            if cfg_strength == 0:
                return fw_fn_uncond(t, x, *args, **kwargs)
            elif cfg_strength == 1:
                return fw_fn_cond(t, x, *args, **kwargs)
            else:
                return cfg_strength * fw_fn_cond(t, x, *args, **kwargs) + (1. - cfg_strength) * fw_fn_uncond(t, x, *args, **kwargs)


    node_ = NeuralODE(fw_fn, solver="euler", sensitivity="adjoint")
    with torch.no_grad():
        x1_flat = x1.view(-1, 3, h, w) if x1 is not None else None
        x0, _ = get_prior_sample(x1_flat, False, super_resolution_factor) # no ddc since mask is explicitly provided outside -- we take care of it below
        if obs_mask is not None and x1 is not None:
            assert super_resolution_factor <= 0, 'cannot have superres and partial observations together!'
            # put the observed x1 in the observed spot:
            x0 = x0*(1-obs_mask) + x1_flat*obs_mask
        traj = node_.trajectory(
            x0,
            t_span=torch.linspace(0, 1, num_int_steps, device=device),
        )
        traj = traj.clip(-1,1).cpu()
        traj = traj / 2 + 0.5
        traj_full = traj.view(num_int_steps, -1, 3, h, w)
        traj = traj[-1, :].view([-1, 3, h, w])


    im_conds = im_conds.cpu()

    # plot bg
    bg_norm = im_conds.view(-1, c_cond, h, w)
    if use_cfg:
        bg_norm = bg_norm[:, :1]
    bg_norm = bg_norm.repeat(1, 3, 1, 1)
    bg_norm = torch.from_numpy(((bg_norm - bg_norm.min()) / (bg_norm.max() - bg_norm.min())).cpu().detach().numpy())
    if savedir is not None:
        save_image(bg_norm, savedir + f"{net_}_bg_step_{step}.png", nrow=num_rows_plot)

    # plot cells
    traj_norm = torch.from_numpy(
        plot_topo_cells(traj.cpu().permute(0,2,3,1).numpy())
    ).permute(0,3,1,2) / 255.
    if savedir is not None:
        save_image(traj_norm, savedir + f"{net_}_cells_step_{step}.png", nrow=num_rows_plot)

    # plot cells superimposed on bg:
    traj_and_bg = torch.cat([im_conds.view(-1, c_cond, h, w)[:, :1] / 2 + 0.5, traj], 1)
    traj_and_bg_norm = torch.from_numpy(
        plot_topo_cells(traj_and_bg.cpu().permute(0, 2, 3, 1).numpy(), bg_norm=0.5)
    ).permute(0, 3, 1, 2) / 255.
    if savedir is not None:
        save_image(traj_and_bg_norm, savedir + f"{net_}_cells_and_bg_step_{step}.png", nrow=num_rows_plot)

    model.train()
    if not return_whole_traj:
        return traj_and_bg
    else:
        return traj_full

def generate_samples_shell(model, parallel, y, savedir, step, net_="normal", bs=8, shape=(1, 2048), n_int_steps=100):
    """

    Parameters
    ----------
    model:
        represents the neural network that we want to generate samples from
    parallel: bool
        represents the parallel training flag. Torchdyn only runs on 1 GPU, we need to send the models from several GPUs to 1 GPU.
    savedir: str
        represents the path where we want to save the generated images
    step: int
        represents the current step of training
    im_conds: torch.Tensor: (num_rows, num_cols, C, H, W)
        represents the conditioning images. The shape (num_rows, num_cols) will be the subplot shape
    """
    model.eval()

    model_ = copy.deepcopy(model)
    model_.forward = partial(model_.forward, y=y)
    if parallel:
        # Send the models from GPU to CPU for inference with NeuralODE from Torchdyn
        model_ = model_.module.to(device)

    node_ = NeuralODE(model_, solver="euler", sensitivity="adjoint")
    with torch.no_grad():
        traj = node_.trajectory(
            torch.randn(bs, *shape, device=device),
            t_span=torch.linspace(0, 1, n_int_steps, device=device),
        )
        traj = traj[-1, :].view([-1, *shape])

    return traj.to('cpu')


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay)
        )


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x, y


def normalize_channelwise(arr, ignore_zeros=None, do_simple_normalization=False):
    if do_simple_normalization:
        arr = arr.astype(float)
        arr = arr / 65535. * 2 - 1
        # arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
        return arr

    if ignore_zeros is None:
        ignore_zeros = tuple(True for _ in range(arr.shape[-1]))
    elif not isinstance(ignore_zeros, tuple):
        ignore_zeros = (ignore_zeros for _ in range(arr.shape[-1]))

    for i in range(arr.shape[-1]):
        if ignore_zeros[i]:
            mask = arr[...,i] != 0
            if mask.any(): # if we have any nonzero entries
                min = arr[...,i][mask].min()
            else:
                min = 0.
        else:
            min = arr[...,i].min()

        arr[..., i] = np.maximum(arr[..., i] - min, 0)  # keep zeros at zero
        arr[..., i] /= (arr[..., i].max() + 1e-6) # avoid division by zero error
        arr[..., i] = arr[..., i] * 2 - 1

    return arr

def normalize_channelwise_per_chip(arr, min_val_for_norm=np.array([[0.]]), max_val_for_norm=np.array([[4095.]])):
    arr = arr.astype(float)
    low = min_val_for_norm[None, None, ...]
    high = max_val_for_norm[None, None, ...]
    # map to [0,1] -- zeros were ignored in min, so we might have some vals below zero that were outside the cell bbox
    arr_norm = (arr - low) / (high - low)
    # sanity check: min should be at least 0, or all values within the possible crop are 0. max should be at most 1
    assert (np.nanmin(np.where(arr > (0+(1e-6)), arr_norm, np.nan), axis=(0,1)) >= (0-1e-6)).all() or (arr==0).all((0,1)).any(), 'normalization did not give expected results: lowest < 0'
    assert (arr_norm.max((0,1)) <= (1+(1e-6))).all(), 'normalization did not give expected results: highest > 1'
    arr_norm = np.clip(arr_norm, 0, 1)  # clip to [0,1]
    arr_norm = arr_norm * 2 - 1  #[-1,1]
    return arr_norm


def plot_topo_cells(arr, make_plot=False, bg_norm=0.7, channelwise_norm=False,
                    norm_func=lambda x: normalize_channelwise(x)*0.5 + 0.5):
    if len(arr.shape) == 3: # one image:
        return plot_topo_cells_one_im(arr,
                                      make_plot=make_plot, bg_norm=bg_norm,
                                      channelwise_norm=channelwise_norm, norm_func=norm_func)
    elif len(arr.shape) == 4: # batch of images:
        out = []
        for i in range(arr.shape[0]):
            out.append(plot_topo_cells_one_im(arr[i],
                                      make_plot=make_plot, bg_norm=bg_norm,
                                      channelwise_norm=channelwise_norm, norm_func=norm_func)
                       )
        out = np.stack(out, 0)
        return out

def plot_topo_cells_one_im(arr, make_plot=False, bg_norm=0.4, channelwise_norm=False,
                           norm_func=lambda x: normalize_channelwise(x) * 0.5 + 0.5):
    if isinstance(norm_func, str):
        if norm_func == 'min-max':
            norm_func = lambda arr: (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
        elif norm_func == 'z-norm':
            norm_func = lambda arr: (arr - np.mean(arr)) / np.std(arr) / 3
        else:
            raise NotImplementedError('norm mode not recognized')
    elif isinstance(norm_func, Callable):
        norm_func = norm_func
    else:
        raise ValueError('norm_func should be a string or callable normalization function')

    arr = arr.astype(float)
    if not channelwise_norm:
        arr = norm_func(arr)
    else:
        for c in range(arr.shape[-1]):
            arr[..., c] = norm_func(arr[..., c])

    # first channel is topo
    to_plot = np.zeros((*arr.shape[:2], 3))
    if arr.shape[-1] == 4:
        to_plot += arr[..., 0:1] * bg_norm  # dim this down a bit for visualization purposes
        to_plot += arr[..., 1:]
    elif arr.shape[-1] == 3:
        to_plot += arr
    else:
        raise ValueError('expected 3 or 4 channels in the input array')
    to_plot = np.minimum(np.round(to_plot * 255).astype(int), 255)
    if make_plot:
        plt.figure(figsize=(10, 10))
        plt.imshow(to_plot)
        plt.show()
    return to_plot







def plot_topo_cells_old(arr, make_plot=False, bg_multiplier=0.7):
    if len(arr.shape) == 3:
        # unbatched input
        low = np.min(arr)
        high = np.max(arr)
    else:
        # batched input -- calculation per batch element
        low = np.min(arr, axis=(1,2,3), keepdims=True)
        high = np.max(arr, axis=(1,2,3), keepdims=True)

    arr = (arr - low) / (high - low)  # from [-1,1] to [0,1] -- used to be (arr + 1) / 2

    # first channel is topo
    to_plot = np.zeros((*arr.shape[:-1], 3))
    if arr.shape[-1] ==4:
        to_plot += arr[..., 0:1] * bg_multiplier  # dim this down a bit for visualization purposes
        to_plot += arr[..., 1:]
    elif arr.shape[-1] == 3:
        to_plot += arr
    else:
        raise ValueError('expected 3 or 4 channels in the input array')
    # correct for the background oversaturation:
    to_plot = to_plot - np.mean(arr[..., 0:1] * bg_multiplier, axis=(1,2,3), keepdims=True)
    to_plot = np.maximum(np.minimum(np.round(to_plot * 255).astype(int), 255), 0)
    if make_plot:
        plt.figure(figsize=(10, 10))
        plt.imshow(to_plot)
        plt.show()
    return to_plot




def mask_approx_yap(arr):
    # multiply the values in the YAP channel (1) with the values in the pholloiding channel 2 to get rid of the overall
    # high illumination outside of the cells. Important: normalize afterwards!

    yap = arr[..., 1]
    pholl = arr[..., 2]
    yap *= pholl
    arr[..., 1] = yap
    return arr



def save_image(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    fp: Union[str, pathlib.Path, BinaryIO],
    format: Optional[str] = None,
    nrow: int = 3,
) -> None:
    """
    Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        fp (string or file object): A filename or a file object
        format(Optional):  If omitted, the format to use is determined from the filename extension.
            If a file object was used instead of a filename, this parameter should always be used.
        num_rows: number of rows in the grid.
    """

    # if not torch.jit.is_scripting() and not torch.jit.is_tracing():
    #     _log_api_usage_once(save_image)
    # grid = make_grid(tensor, **kwargs)
    # # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
    # ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    # im = Image.fromarray(ndarr)
    # im.save(fp, format=format)
    ncol = np.ceil(tensor.shape[0] / nrow).astype(int)
    h, w = tensor.shape[-2:]
    grid = np.zeros(shape=(nrow * (h+3), ncol * (w+3), 3))
    for i in range(tensor.shape[0]):
        row = i // ncol
        col = i % ncol
        grid[row*(h+3):row*(h+3)+h, col*(w+3):col*(w+3)+w] = tensor[i].permute(1, 2, 0).numpy()
    plt.imsave(fp, grid, format=format)
    plt.close()



def resize(im, num_blocks):
    """
    interpolate im size to nearest larger integer divisible by 2^(self.num_channel_mults.__len__())
    :param im: image to pad
    :return: padded image, intepolation factor
    """
    size = im.shape[-1]
    div_by = 2 ** num_blocks
    desired_size = np.ceil(size / div_by) * div_by
    out = F.interpolate(im, int(desired_size), mode='bilinear')
    return out

def undo_resize(original_im, im):
    """
    interpolate im size to nearest larger integer divisible by 2^(self.num_channel_mults.__len__())
    :param im: image to pad
    :return: padded image, intepolation factor
    """
    size = original_im.shape[-1]
    out = F.interpolate(im, size, mode='bilinear')
    return out

def lazy_conv_nd(dims, *args, **kwargs):
    """Create a 1D, 2D, or 3D convolution module."""
    if dims == 1:
        return nn.LazyConv1d(*args, **kwargs)
    elif dims == 2:
        return nn.LazyConv2d(*args, **kwargs)
    elif dims == 3:
        return nn.LazyConv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")



def prepare_cfg_conditioning(cond, prob_no_guidance=0.):
    assert 0. <= prob_no_guidance <= 1.
    cond_add_ch = torch.cat([cond,
                             torch.ones(size=(cond.shape[0], 1, *cond.shape[2:])).to(cond)
                             ], dim=1)  # we have 1 is the conditioning signal is present, 0 otherwise
    remove_guidance = torch.rand(cond.shape[0]) < prob_no_guidance
    cond_add_ch[remove_guidance] = 0.
    return cond_add_ch





def get_prior_sample(x1, use_ddc=False, super_resolution_factor=0):
    if not use_ddc and super_resolution_factor <= 0:
        return torch.randn_like(x1), None

    elif use_ddc:
        assert super_resolution_factor <= 0
        # we are using data-dependent coupling:
        # - at least 1 channels should not be observed
        # - it can be that all three channels are not observed
        num_unobserved_channels = np.random.choice(np.arange(1,4), size=x1.shape[0],
                                                   replace=True,
                                                   p=np.array([0.1, 0.1, 0.8])
                                                   ) # for each batch element, how many channels are not observed?

        random_indices = torch.argsort(torch.rand(x1.shape[0], x1.shape[1]), dim=1)  # randomly shuffled indices in [0,1,2]
        mask = random_indices < torch.from_numpy(num_unobserved_channels).unsqueeze(1)  # mask of unobserved channels
        mask = mask[..., None, None].to(x1.device).expand_as(x1)  # repeat mask for all spatial dimensions

        assert (
                torch.all(mask, dim=(-1, -2)) == torch.any(mask, dim=(-1, -2))
        ).all(), 'mask should have the same value within channels'

        x0 = torch.randn_like(x1)
        x0[mask] = x1[mask]

        return x0, mask.to(x1)

    elif super_resolution_factor > 0:
        assert not use_ddc and super_resolution_factor > 1
        size = x1.shape[-2:]
        upsample = Resize([int(np.round(s)) for s in size])
        downsample = Resize([int(np.round(s / super_resolution_factor)) for s in size])
        x0 = upsample(downsample(x1))
        x0_slightly_noisy = x0 + torch.randn_like(x0) * torch.std(x0) * 0.25  # std of gaussian noise is 10% of std of data
        return x0_slightly_noisy, None
    else:
        raise NotImplementedError('not implemented')




