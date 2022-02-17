import h5py
from matplotlib import pyplot as plt
import numpy as np
from skimage.transform import iradon


def shift_cut_sinogram(sinogram, img_size: int, circle: bool):
    target_size = img_size if circle else int(np.ceil(img_size*np.sqrt(2)))
    if sinogram.shape[0] < target_size:
        raise NotImplementedError("Sino must be bigger than target")
    if sinogram.shape[0] % 2 == 0 and target_size % 2 == 1:
        fftsino = np.fft.fft(sinogram, axis=0)
        N = fftsino.shape[0]
        shiftmap = np.arange(N)
        shiftmap = np.mod((shiftmap + N/2), N) - N/2
        shiftmap = np.exp(-2j*np.pi*1/N/2*shiftmap)
        shiftedfftsino = fftsino*shiftmap[:, None]
        shiftedsino = np.real(np.fft.ifft(shiftedfftsino, axis=0))
        shiftedsino = shiftedsino[1:]
        fromto = (sinogram.shape[0] - target_size)//2
        cutsino = shiftedsino[fromto:-fromto]
        return cutsino

    raise NotImplementedError("Only implemented for even sino and odd target.")


def main():
    dataset = h5py.File(
        'equidist_sparse8_spokeifft_val.h5',
        'r',
        libver='latest',
        swmr=True)

    sino = dataset['0']['sinoMag'][()]
    cutsino = shift_cut_sinogram(sino, 256, False)
    reco = iradon(cutsino, theta=-np.linspace(0, 180,
                  num=sino.shape[1], endpoint=False), circle=False)
    plt.imshow(reco, vmin=0, vmax=0.4)
    plt.figure()
    plt.imshow(dataset['0']['imgMag'][()], vmin=0, vmax=0.4)
    plt.figure()
    plt.imshow(np.abs(reco-dataset['0']['imgMag'][()]), vmin=0, vmax=0.4)
    plt.show()


if __name__ == '__main__':
    main()
