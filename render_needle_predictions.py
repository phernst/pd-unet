import json
from os.path import join as pjoin

import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from tqdm import tqdm

from needle_simulations import generate_needle_simulations
from test_pd_unet import render_needle_predictions as rnp_pd_unet
from test_pd_orig import render_needle_predictions as rnp_pd_orig
from test_reco_unet import render_needle_predictions as rnp_reco_unet
from test_sino_unet import render_needle_predictions as rnp_sino_unet
from utilities.transforms import hu2mu


def render_needle_predictions():
    with open('train_valid.json', 'r', encoding='utf-8') as json_file:
        test_files = json.load(json_file)['test_files']

    parfan = 'fan'
    subtype = 'sparse'
    subnum = 16

    for test_file in tqdm(test_files):
        groundtruth = generate_needle_simulations(test_file)
        rnp_pd_unet(parfan, subtype, subnum, groundtruth, test_file)
        rnp_pd_orig(parfan, subtype, subnum, groundtruth, test_file)
        rnp_reco_unet(parfan, subtype, subnum, groundtruth, test_file)
        rnp_sino_unet(parfan, subtype, subnum, groundtruth, test_file)


def calculate_roi_metrics():
    with open('train_valid.json', 'r', encoding='utf-8') as json_file:
        test_files = json.load(json_file)['test_files']

    predictions_path = 'qualitative_needle/fan_sparse16'

    roi_slice_x = slice(89 - 16, 89 + 16)
    roi_slice_y = slice(121 - 16, 121 + 16)

    all_prediction_types = {
        'pd_unet', 'pd_orig', 'reco_unet', 'sino_unet',
        'under', 'bilin_reco'
    }
    all_rmses = {img_type: [] for img_type in all_prediction_types}
    all_ssims = {img_type: [] for img_type in all_prediction_types}

    for test_file in test_files:
        roi = {
            img_type: np.load(
                pjoin(predictions_path, f'{test_file}_{img_type}.npy')
            ).squeeze()[roi_slice_y, roi_slice_x]
            for img_type in all_prediction_types | {'gt'}
        }

        max_mu = hu2mu(roi['gt']).max()
        norm_roi = {
            img_type: hu2mu(img)/max_mu for img_type, img in roi.items()
        }

        test_file_ssim = {
            img_type: compare_ssim(
                norm_roi['gt'],
                norm_roi[img_type],
                data_range=1,
                full=True,
            )
            for img_type in all_prediction_types
        }

        for img_type, values in test_file_ssim.items():
            np.save(pjoin(predictions_path, f'{test_file}_{img_type}_map'), values[1])

        test_file_rmse = {
            img_type: np.sqrt(np.mean((roi['gt'] - roi[img_type])**2))
            for img_type in all_prediction_types
        }

        all_rmses = {
            img_type: values + [test_file_rmse[img_type]]
            for (img_type, values) in all_rmses.items()
        }

        all_ssims = {
            img_type: values + [test_file_ssim[img_type][0]]
            for (img_type, values) in all_ssims.items()
        }


    mean_std_rmse = {img_type: (np.mean(all_rmses[img_type]), np.std(all_rmses[img_type])) for img_type in all_rmses}
    mean_std_ssim = {img_type: (np.mean(all_ssims[img_type]), np.std(all_ssims[img_type])) for img_type in all_ssims}

    for img_type in ['under', 'bilin_reco', 'sino_unet', 'reco_unet', 'pd_orig', 'pd_unet']:
        print(f'{img_type} & {mean_std_ssim[img_type][0]:.3f}\\textpm'
              f'{mean_std_ssim[img_type][1]:.3f} & '
              f'{mean_std_rmse[img_type][0]:.0f}\\textpm'
              f'{mean_std_rmse[img_type][1]:.0f} \\\\')


def create_figure_images():
    all_prediction_types = {
        'pd_unet', 'pd_orig', 'reco_unet', 'sino_unet',
        'under', 'bilin_reco'
    }

    predictions_path = 'qualitative_needle/fan_sparse16'
    test_file = 'ABD_LYMPH_049.nii.gz'
    out_path = 'qualitative_needle/figures'

    image = {
        img_type: np.load(
            pjoin(predictions_path, f'{test_file}_{img_type}.npy')
        ).squeeze()
        for img_type in all_prediction_types | {'gt'}
    }

    # maxdiff = np.max([
    #     np.max(np.abs(image['gt'] - image[img_type]))
    #     for img_type in all_prediction_types
    # ])
    # print(maxdiff)

    diff_imgs = {
        img_type: np.abs(image['gt'] - image[img_type])
        for img_type in all_prediction_types
    }

    gt = np.load(
        pjoin(predictions_path, f'{test_file}_gt.npy')
    ).squeeze()
    gt = np.clip(gt, -1000, 1000)
    gt += 1000
    gt = (gt/np.ptp(gt)*255).astype(np.uint8)
    cv2.imwrite(pjoin(out_path, f'{test_file}_gt.png'), gt)

    for img_type in all_prediction_types:
        print(img_type)
        # print(f'rmse: {np.sqrt(np.mean((image["gt"] - image[img_type])**2))}')
        img = image[img_type]
        img = np.clip(img, -1000, 1000)
        img += 1000
        img = (img/np.ptp(img)*255).astype(np.uint8)
        cv2.imwrite(pjoin(out_path, f'{test_file}_{img_type}.png'), img)

        diff_img = (np.clip(diff_imgs[img_type], 0, 1000)/1000*255).astype(np.uint8)
        diff_img = cv2.applyColorMap(diff_img, cv2.COLORMAP_HOT)
        cv2.imwrite(pjoin(out_path, f'{test_file}_{img_type}_diff.png'), diff_img)

        ssim_value, ssimmap = compare_ssim(
            image['gt'],
            image[img_type],
            data_range=np.ptp(image[img_type]),
            full=True,
        )
        print(f'ssim: {ssim_value}')
        ssimmap = ((ssimmap-ssimmap.min())/np.ptp(ssimmap)*255).astype(np.uint8)
        cv2.imwrite(pjoin(out_path, f'{test_file}_{img_type}_map.png'), ssimmap)


if __name__ == '__main__':
    calculate_roi_metrics()


# needle tip: 89, 121
