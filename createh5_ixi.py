import os
import numpy as np
# from scipy.misc import imresize
import h5py
from tqdm import tqdm
import scipy.io as sio
import glob
import skimage.transform as sktrans
import sys
from pynufft import NUFFT as NUFFT_cpu
from utilities.pyn.sampler import createSampling

source_MATs = list(glob.iglob(r'Datasets/IXI/MATs/IXI-T1/HH/Slice60to90/Varden1D30SameMask/test/*.mat'))
output_file = r'Datasets/IXI/H5/IXI_T1_HH_Slice60to90/Xequidist_sparse8_spokeifft_test.h5'
sp = 512
baseresolution = 256
interpolationSize4NUFFT = 6
isGolden = False

# undersampling
undersampling_type = "sparse"  # limitedangle
firstNSpoke = 128  # to be used for limitedangle
everyNSpoke = 8  # to be used for sparse
interpOrder = 1

# Prepare the h5 file
if(not os.path.isfile(output_file)):
    h5 = h5py.File(output_file, "w")
    offset = 0
else:
    print('The HDF5 file specified already exists. Do you want to append to it? If no, then the existing file will be overwritten')
    answer = input("Enter Y/N: ")
    if answer == 'Y' or answer == 'y':
        h5 = h5py.File(output_file, "a")
        offset = len(h5)
    else:
        h5 = h5py.File(output_file, "w")
        offset = 0

# Prepare the nufft operators
omT, dcfT, angle_inc = createSampling(np.zeros((baseresolution, baseresolution)), 60, isGolden=isGolden)
fully_angle = omT[1]
fully_dcf = dcfT[1]
nSpokes_fully = fully_angle.shape[0] // (baseresolution*2)

# With Fully
NufftObj = NUFFT_cpu()
Nd = (baseresolution, baseresolution)  # image size
Kd = (baseresolution*2, baseresolution*2)  # k-space size
Jd = (interpolationSize4NUFFT, interpolationSize4NUFFT)  # interpolation size
NufftObj.plan(fully_angle, Nd, Kd, Jd)

# With Fully one spoke
NufftObjSpk = NUFFT_cpu()
NdSpk = (baseresolution*2,)  # image size - the spokes are already twice the resolution. this will make nufft adjoint to return of the same size (twice the base res) - oversampling will avoid circle like artefact
KdSpk = (baseresolution*4,)  # k-space size - further oversampling will avoid wrap-in artefacts
JdSpk = (interpolationSize4NUFFT,)  # interpolation size
NufftObjSpk.plan(np.expand_dims(fully_angle[0:nSpokes_fully, 1], -1), NdSpk, KdSpk, JdSpk)


def generateSino(radKSP, useNUFFTAdj=True):
    sinoMag = np.zeros((baseresolution*2, nSpokes_fully))
    sinoPhs = np.zeros((baseresolution*2, nSpokes_fully))
    for i in range(nSpokes_fully):
        lineKSP = radKSP[i*nSpokes_fully:(i+1)*nSpokes_fully]
        if useNUFFTAdj:
            lineImg = NufftObjSpk.adjoint(lineKSP)
        else:
            lineImg = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(lineKSP)))
        sinoMag[:, i] = np.abs(lineImg)
        sinoPhs[:, i] = np.angle(lineImg)
    theta = np.linspace(0, sinoMag.shape[1], num=sinoMag.shape[1], endpoint=False) * angle_inc
    imgMag = sktrans.iradon(sinoMag, theta=-theta, output_size=baseresolution, circle=False)
    imgPhs = sktrans.iradon(sinoPhs, theta=-theta, output_size=baseresolution, circle=False)
    return (sinoMag, sinoPhs), (imgMag, imgPhs)


def undersampleSino(sino, sampling='sparse', everyNSpoke=None, firstNSpoke=None):
    fullyAngle = np.linspace(0, sino.shape[1], num=sino.shape[1], endpoint=False) * angle_inc
    if sampling == 'sparse' and everyNSpoke is not None:
        underSino = sino[:, ::everyNSpoke]
        underAngle = fullyAngle[::everyNSpoke]
    elif sampling == 'limitedangle' and firstNSpoke is not None:
        underSino = sino[:, :firstNSpoke]
        underAngle = fullyAngle[:firstNSpoke]
    else:
        sys.exit('Invalid Sampling or Insufficient Sampling Parameters')
    underImg = sktrans.iradon(underSino, theta=-underAngle, output_size=baseresolution, circle=False)
    return underSino, underAngle, underImg


def upsampleSino(underSino, underAngle, upShape, interpOrder=1, unsortSino=True):
    underAngleSortInd = np.argsort(underAngle % 360)
    sinoSorted = underSino[:, underAngleSortInd]
    # sinoUp = imresize(sinoSorted, upShape, interp=interpAlgo)
    sinoUp = sktrans.resize(sinoSorted, upShape, order=interpOrder)
    thetaUp = np.linspace(0, upShape[1], num=upShape[1], endpoint=False) * angle_inc
    thetaUpSortInd = np.argsort(thetaUp % 360)
    if unsortSino:
        thetaUpUnSortInd = np.argsort(thetaUpSortInd)
        sinoUp = sinoUp[:, thetaUpUnSortInd]
    else:
        thetaUp = thetaUp[thetaUpSortInd]
    underImgUp = sktrans.iradon(sinoUp, theta=-thetaUp, output_size=baseresolution, circle=False)
    return sinoUp, underImgUp


# Do the actual job
with tqdm(total=len(source_MATs)) as pbar:
    for i, mat_path in enumerate(source_MATs, 0):
        h5ds = h5.create_group(str(i+offset))
        mat = sio.loadmat(mat_path)
        h5ds.create_dataset('fileName', data=mat['fileName'].astype('S'))  # the channel dim is not gonna be needed in our case (gonna be taken care of by the main code)
        h5ds.create_dataset('subjectName', data=mat['subjectName'].astype('S'))  # the channel dim is not gonna be needed in our case (gonna be taken care of by the main code)
        gaKSP = NufftObj.forward(mat['fully'].squeeze())

        (sinoMag, sinoPhs), (imgMag, imgPhs) = generateSino(gaKSP, useNUFFTAdj=False)

        underSinoMag, underAngle, underImgMag = undersampleSino(sinoMag, sampling=undersampling_type, everyNSpoke=everyNSpoke, firstNSpoke=firstNSpoke)
        underSinoPhs, _, underImgPhs = undersampleSino(sinoPhs, sampling=undersampling_type, everyNSpoke=everyNSpoke, firstNSpoke=firstNSpoke)

        sinoMagUp, underImgMagUp = upsampleSino(underSinoMag, underAngle, sinoMag.shape, interpOrder=interpOrder, unsortSino=True)
        sinoPhsUp, underImgPhsUp = upsampleSino(underSinoPhs, underAngle, sinoPhs.shape, interpOrder=interpOrder, unsortSino=True)

        h5ds.create_dataset('sinoMag', data=sinoMag)
        h5ds.create_dataset('sinoPhs', data=sinoPhs)
        h5ds.create_dataset('imgMag', data=imgMag)
        h5ds.create_dataset('imgPhs', data=imgPhs)

        h5ds.create_dataset('underSinoMag', data=underSinoMag)
        h5ds.create_dataset('underSinoPhs', data=underSinoPhs)
        h5ds.create_dataset('underImgMag', data=underImgMag)
        h5ds.create_dataset('underImgPhs', data=underImgPhs)

        h5ds.create_dataset('underAngle', data=underAngle)

        h5ds.create_dataset('sinoMagUp', data=sinoMagUp)
        h5ds.create_dataset('sinoPhsUp', data=sinoPhsUp)
        h5ds.create_dataset('underImgMagUp', data=underImgMagUp)
        h5ds.create_dataset('underImgPhsUp', data=underImgPhsUp)

        pbar.update(1)
