from os.path import join as pjoin

import nibabel as nib
from skimage.transform import downscale_local_mean
import torch
from utilities.transforms import hu2mu, mu2hu


def generate_needle_simulations(abd_filename: str):
    abd_directory = 'big_data'
    abd_data = nib.load(pjoin(abd_directory, abd_filename)).get_fdata()
    abd_data = abd_data[..., abd_data.shape[-1]//2 - 100:abd_data.shape[-1]//2 + 100]
    abd_data = mu2hu(abd_data)

    needle_directory = 'needles'
    needle_filename = 'Needle2_Pos1_11.nii.gz'
    needle_data = nib.load(pjoin(needle_directory, needle_filename)).get_fdata()

    print(f'{abd_data.shape=}, {needle_data.shape=}')

    abd_data = abd_data[..., -min(needle_data.shape[-1], abd_data.shape[-1]):]
    needle_data = needle_data[..., -min(needle_data.shape[-1], abd_data.shape[-1]):][..., ::-1]
    combined_data = hu2mu(abd_data) + hu2mu(needle_data)
    combined_data = combined_data.transpose(1, 0, 2)
    combined_data = downscale_local_mean(combined_data, (2, 2, 1))

    gt_index = 60
    return torch.from_numpy(combined_data[..., gt_index])[None, None]
