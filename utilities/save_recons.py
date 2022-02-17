import os
from math import sqrt
import numpy as np
import scipy.io as sio
import nibabel as nib
from skimage.metrics import structural_similarity as compare_ssim


def FileSave(data, file_path):
    """Save a NIFTI file using given file path from an array
    Using: NiBabel"""
    if np.iscomplex(data).any():
        data = abs(data)
    nii = nib.Nifti1Image(data, np.eye(4))
    nib.save(nii, file_path)


def save_sinograms(prediction: np.array, bilin: np.array, path4output: str):
    if not os.path.exists(path4output):
        os.makedirs(path4output)

    FileSave(prediction, os.path.join(path4output, "prediction_sino.nii"))
    FileSave(bilin, os.path.join(path4output, "bilin_sino.nii"))


def ValidateNStore(Out, Ref, Undersampled, path4output, getSSIMMap=True, doNorm=True):
    # Creating all the necesaary file names
    if not os.path.exists(path4output):
        os.makedirs(path4output)
    accuracyFileName = os.path.join(path4output, 'accuracy.txt')
    ssimMapFileName = os.path.join(path4output, 'ssimmap.mat')
    fullySampledVolFileName = os.path.join(path4output, 'fully.nii')
    underSampledVolFileName = os.path.join(path4output, 'under.nii')
    outputVolFileName = os.path.join(path4output, 'recon.nii')
    ssimMapUndersampledFileName = os.path.join(path4output, 'ssimmapOfUndersampled.mat')
    accuracyUndersampledFileName = os.path.join(path4output, 'accuracyOfUndersampled.txt')
    improvementFileName = os.path.join(path4output, 'improvement.txt')

    if doNorm:
        Out = Out/Out.max()
        Ref = Ref/Ref.max()
        Undersampled = Undersampled/Undersampled.max()

    FileSave(Ref, fullySampledVolFileName)
    FileSave(Undersampled, underSampledVolFileName)
    FileSave(Out, outputVolFileName)

    # Calculate Accuracy of the Output
    errorPercent = np.mean(Out != Ref) * 100
    mse = ((Out - Ref) ** 2).mean()

    # Calculate Accuracy of the Undersampled Image
    errorPercentUndersampled = np.mean(Undersampled != Ref) * 100
    mseUndersampled = ((Undersampled - Ref) ** 2).mean()

    if(np.iscomplex(Out).any()):
        rmse = sqrt(abs(mse))
        rmseUndersampled = sqrt(abs(mseUndersampled))
        Ref4SSIM = abs(Ref).astype('float32')
        Out4SSIM = abs(Out).astype('float32')
        Under4SSIM = abs(Undersampled).astype('float32')
    else:
        rmse = sqrt(mse)
        rmseUndersampled = sqrt(mseUndersampled)
        Ref4SSIM = Ref.astype('float32')
        Out4SSIM = Out.astype('float32')
        Under4SSIM = Undersampled.astype('float32')

    # Calculate SSIM of Output and save SSIM as well as aaccuracy
    if(getSSIMMap):
        ssim, ssimMAP = compare_ssim(Ref4SSIM, Out4SSIM, data_range=Out4SSIM.max() - Out4SSIM.min(), multichannel=True, full=True)  # with Map
        sio.savemat(ssimMapFileName, {'ssimMAP': ssimMAP})
    else:
        ssim = compare_ssim(Ref4SSIM, Out4SSIM, data_range=Out4SSIM.max() - Out4SSIM.min(), multichannel=True)  # without Map

    file = open(accuracyFileName, "w")
    file.write("Error Percent: " + str(errorPercent))
    file.write("\r\nMSE: " + str(mse))
    file.write("\r\nRMSE: " + str(rmse))
    file.write("\r\nMean SSIM: " + str(ssim))
    file.close()

    # Calculate SSIM of Undersampled and save SSIM as well as aaccuracy
    if(getSSIMMap):
        ssimUndersampled, ssimUndersampledMAP = compare_ssim(Ref4SSIM, Under4SSIM, data_range=Under4SSIM.max() - Under4SSIM.min(), multichannel=True, full=True)  # 3D, with Map
        sio.savemat(ssimMapUndersampledFileName, {'ssimMAP': ssimUndersampledMAP})
    else:
        ssimUndersampled = compare_ssim(Ref4SSIM, Under4SSIM, data_range=Under4SSIM.max() - Under4SSIM.min(), multichannel=True)  # 3D

    file = open(accuracyUndersampledFileName, "w")
    file.write("Error Percent: " + str(errorPercentUndersampled))
    file.write("\r\nMSE: " + str(mseUndersampled))
    file.write("\r\nRMSE: " + str(rmseUndersampled))
    file.write("\r\nMean SSIM: " + str(ssimUndersampled))
    file.close()

    # Check for Accuracy Improvement
    errorPercentImprovement = errorPercentUndersampled - errorPercent
    mseImprovement = mseUndersampled - mse
    rmseImprovement = rmseUndersampled - rmse
    ssimImprovement = ssim - ssimUndersampled
    file = open(improvementFileName, "w")
    file.write("Error Percent: " + str(errorPercentImprovement))
    file.write("\r\nMSE: " + str(mseImprovement))
    file.write("\r\nRMSE: " + str(rmseImprovement))
    file.write("\r\nMean SSIM: " + str(ssimImprovement))
    file.close()

    return {
        "Img": os.path.basename(path4output),
        "ErrorPercent (Out)": errorPercent,
        "ErrorPercent (Under)": errorPercentUndersampled,
        "ErrorPercent (Improve)": errorPercentImprovement,
        "MSE (Out)": mse,
        "MSE (Under)": mseUndersampled,
        "MSE (Improve)": mseImprovement,
        "RMSE (Out)": rmse,
        "RMSE (Under)": rmseUndersampled,
        "RMSE (Improve)": rmseImprovement,
        "SSIM (Out)": ssim,
        "SSIM (Under)": ssimUndersampled,
        "SSIM (Improve)": ssimImprovement,
    }
