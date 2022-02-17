import os
from math import sqrt
from typing import Optional
import numpy as np
import scipy.io as sio
import nibabel as nib
from skimage.metrics import structural_similarity as compare_ssim


def file_save(data, file_path):
    """Save a NIFTI file using given file path from an array
    Using: NiBabel"""
    if np.iscomplex(data).any():
        data = abs(data)
    nii = nib.Nifti1Image(data, np.eye(4))
    nib.save(nii, file_path)


def save_sinograms(prediction: np.ndarray, bilin: np.ndarray, path4output: str):
    if not os.path.exists(path4output):
        os.makedirs(path4output)

    file_save(prediction, os.path.join(path4output, "prediction_sino.nii"))
    file_save(bilin, os.path.join(path4output, "bilin_sino.nii"))


def validate_consistency(prediction: np.ndarray,
                         reference: np.ndarray,
                         bilinear: Optional[np.ndarray] = None):
    error_percent = np.mean(prediction != reference) * 100
    mse = ((prediction - reference) ** 2).mean()
    rmse = sqrt(mse)
    ref4ssim = reference.astype('float32')
    pred4ssim = prediction.astype('float32')

    ssim = compare_ssim(
        ref4ssim,
        pred4ssim,
        data_range=np.ptp(pred4ssim),
    )

    bilinear = np.zeros_like(reference) if bilinear is None else bilinear
    error_percent_under = np.mean(bilinear != reference) * 100
    mse_under = ((bilinear - reference) ** 2).mean()
    rmse_under = sqrt(mse_under)
    under4ssim = bilinear.astype('float32')

    ssim_under = compare_ssim(
        ref4ssim,
        under4ssim,
        data_range=np.ptp(under4ssim),
    )

    return {
        "ErrorPercent (Sino)": error_percent,
        "ErrorPercent (Bilin-Sino)": error_percent_under,
        "MSE (Sino)": mse,
        "MSE (Bilin-Sino)": mse_under,
        "RMSE (Sino)": rmse,
        "RMSE (Bilin-Sino)": rmse_under,
        "SSIM (Sino)": ssim,
        "SSIM (Bilin-Sino)": ssim_under,
    }


def validate_and_store(out, ref, undersampled, path4output, get_ssim_map=True, do_norm=True):
    # Creating all the necessary file names
    os.makedirs(path4output, exist_ok=True)
    accuracy_file_name = os.path.join(path4output, 'accuracy.txt')
    ssim_map_file_name = os.path.join(path4output, 'ssimmap.mat')
    fully_sampled_vol_file_name = os.path.join(path4output, 'fully.nii')
    under_sampled_vol_file_name = os.path.join(path4output, 'under.nii')
    output_vol_file_name = os.path.join(path4output, 'recon.nii')
    ssim_map_undersampled_file_name = os.path.join(path4output, 'ssimmapOfUndersampled.mat')
    accuracy_undersampled_file_name = os.path.join(path4output, 'accuracyOfUndersampled.txt')
    improvement_file_name = os.path.join(path4output, 'improvement.txt')

    if do_norm:
        out = out/out.max()
        ref = ref/ref.max()
        undersampled = undersampled/undersampled.max()

    file_save(ref, fully_sampled_vol_file_name)
    file_save(undersampled, under_sampled_vol_file_name)
    file_save(out, output_vol_file_name)

    # Calculate Accuracy of the Output
    error_percent = np.mean(out != ref) * 100
    mse = ((out - ref) ** 2).mean()

    # Calculate Accuracy of the Undersampled Image
    error_percent_undersampled = np.mean(undersampled != ref) * 100
    mse_undersampled = ((undersampled - ref) ** 2).mean()

    if np.iscomplex(out).any():
        rmse = sqrt(abs(mse))
        rmse_undersampled = sqrt(abs(mse_undersampled))
        ref4ssim = abs(ref).astype('float32')
        out4ssim = abs(out).astype('float32')
        under4ssim = abs(undersampled).astype('float32')
    else:
        rmse = sqrt(mse)
        rmse_undersampled = sqrt(mse_undersampled)
        ref4ssim = ref.astype('float32')
        out4ssim = out.astype('float32')
        under4ssim = undersampled.astype('float32')

    # Calculate SSIM of Output and save SSIM as well as aaccuracy
    if(get_ssim_map):
        ssim, ssim_map = compare_ssim(ref4ssim, out4ssim, data_range=out4ssim.max() - out4ssim.min(), full=True)  # with Map
        sio.savemat(ssim_map_file_name, {'ssimMAP': ssim_map})
    else:
        ssim = compare_ssim(ref4ssim, out4ssim, data_range=out4ssim.max() - out4ssim.min())  # without Map

    with open(accuracy_file_name, "w", encoding='utf-8') as file:
        file.write("Error Percent: " + str(error_percent))
        file.write("\r\nMSE: " + str(mse))
        file.write("\r\nRMSE: " + str(rmse))
        file.write("\r\nMean SSIM: " + str(ssim))

    # Calculate SSIM of Undersampled and save SSIM as well as aaccuracy
    if(get_ssim_map):
        ssim_undersampled, ssim_undersampled_map = compare_ssim(
            ref4ssim,
            under4ssim,
            data_range=under4ssim.max() - under4ssim.min(),
            full=True)  # 3D, with Map
        sio.savemat(ssim_map_undersampled_file_name, {'ssimMAP': ssim_undersampled_map})
    else:
        ssim_undersampled = compare_ssim(ref4ssim, under4ssim, data_range=under4ssim.max() - under4ssim.min())  # 3D

    with open(accuracy_undersampled_file_name, "w", encoding='utf-8') as file:
        file.write("Error Percent: " + str(error_percent_undersampled))
        file.write("\r\nMSE: " + str(mse_undersampled))
        file.write("\r\nRMSE: " + str(rmse_undersampled))
        file.write("\r\nMean SSIM: " + str(ssim_undersampled))

    # Check for Accuracy Improvement
    error_percent_improvement = error_percent_undersampled - error_percent
    mse_improvement = mse_undersampled - mse
    rmse_improvement = rmse_undersampled - rmse
    ssim_improvement = ssim - ssim_undersampled
    with open(improvement_file_name, "w", encoding='utf-8') as file:
        file.write("Error Percent: " + str(error_percent_improvement))
        file.write("\r\nMSE: " + str(mse_improvement))
        file.write("\r\nRMSE: " + str(rmse_improvement))
        file.write("\r\nMean SSIM: " + str(ssim_improvement))

    return {
        "Img": os.path.basename(path4output),
        "ErrorPercent (Out)": error_percent,
        "ErrorPercent (Under)": error_percent_undersampled,
        "ErrorPercent (Improve)": error_percent_improvement,
        "MSE (Out)": mse,
        "MSE (Under)": mse_undersampled,
        "MSE (Improve)": mse_improvement,
        "RMSE (Out)": rmse,
        "RMSE (Under)": rmse_undersampled,
        "RMSE (Improve)": rmse_improvement,
        "SSIM (Out)": ssim,
        "SSIM (Under)": ssim_undersampled,
        "SSIM (Improve)": ssim_improvement,
    }
