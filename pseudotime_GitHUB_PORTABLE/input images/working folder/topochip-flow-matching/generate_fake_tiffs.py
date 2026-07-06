import numpy as np
import os
import torch
from utils import *
import utils
from unet import ImCondModel, UNetModel_imcond_wrapper
from topodataset import TopoDataSet
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import argparse

if __name__ == '__main__':

    # fixed params for generation experiments:
    weight_path = 'model_weights/otcfm_topo_weights_step_145000.pt'  # 'model_weights/otcfm_topo_weights_step_39000.pt'
    stain_names = ['background', 'YAP', 'Actin', 'DAPI']

    inject_imcond = True
    use_cfg = True  # classifier free guidance
    key = 'ema_model'

    #CLI for variable args:
    parser = argparse.ArgumentParser(description='Generate fake tiff images from a trained model.')
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of samples to generate.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size.')
    parser.add_argument('--cfg_strength', type=float, default=1., help='w of cfg')
    args = parser.parse_args()
    num_samples_to_make = args.num_samples
    cfg_strength = args.cfg_strength
    bs = args.batch_size


    spath = f'generated_images_cfg_{cfg_strength}/'

    #### start loading the required stuff and generating the images ####


    # Load the model
    h = w = 300
    num_channel = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not inject_imcond:
        model = ImCondModel(
            dim=(3, h, w),  # note: in local test run we had this worng (dim was at (..., 32, 32). which effectively means we were applying attention after only one time downsampling (att. resolutions was 16)
            num_res_blocks=2,
            num_channels=num_channel,
            channel_mult=[1, 1, 2, 2, 2],
            num_heads=4,
            num_head_channels=-1,
            attention_ds=(2**5,),
            dropout=0.1,
            use_cfg=use_cfg
        ).to(
            device
        )  # new dropout + bs of 128
    else:
        model = UNetModel_imcond_wrapper(
            dim=(3, h, w),
            num_res_blocks=2,
            num_channels=num_channel,
            channel_mult=[1, 1, 2, 2, 2],
            num_heads=4,
            num_head_channels=-1,
            attention_ds=(2**5,),
            dropout=0.1,
            num_cond_im_channels=2 if use_cfg else 1
        ).to(device)

    checkpoint = torch.load(weight_path)
    model.load_state_dict(checkpoint[key])
    model.to(device)


    num_workers = 12
    downsampling_factor = 1
    limit_num_data = np.inf
    crop_size = 300
    dataset = TopoDataSet(
        dir='data/',
        downsampling_factor=downsampling_factor,
        limit_num_data=limit_num_data,
        crop_size=crop_size,
        return_filename=True
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )


    with tqdm(enumerate(dataloader), total=int(np.ceil(num_samples_to_make / bs)), dynamic_ncols=True) as pbar:
        for counter, (img, bg, filenames) in pbar:
            if counter * bs >= num_samples_to_make:
                break
            if use_cfg:
                bg = utils.prepare_cfg_conditioning(bg, prob_no_guidance=0.)  # no prob of removing guidance here!
            bg = bg[None, ...].to(device)  # add leading dim for the generation function
            bg_and_img = generate_samples_topo(model, False, None, 0, 'test', im_conds=bg,
                                               use_cfg=use_cfg, cfg_strength=cfg_strength, x1=img.to(device))

            for b in range(bg_and_img.shape[0]):
                filename = filenames[b]
                for c in range(bg_and_img.shape[1]):
                    img_c = bg_and_img[b, c]  # shape (h,w)
                    img_c = img_c.cpu().detach().numpy()
                    img_c = (img_c * 255).astype(np.uint8)
                    img_c = Image.fromarray(img_c)
                    chip_folder = filename[filename.find('Chip'):filename.find('Chip') + 11]
                    obj_num = filename[filename.find('NucleiNr') + 8:filename.find('NucleiNr') + 12]
                    fname_save = f'obj{obj_num}_stain{stain_names[c]}_fake.tiff'
                    if not os.path.exists(os.path.join(spath, chip_folder + '_fake')):
                        os.makedirs(os.path.join(spath, chip_folder + '_fake'))
                    img_c.save(os.path.join(spath, chip_folder + '_fake', fname_save))
    print('done')




