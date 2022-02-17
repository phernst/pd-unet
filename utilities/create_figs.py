from os.path import join as pjoin

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from skimage.metrics import structural_similarity as ssim


def min_max(im):
    return (im-im.min()) / (im.max()-im.min())


def axes_clean(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

###################################################################################


# #MRI: IXI
root_path = pjoin('qualitative', "fan_sparse16")
showfiles = ["59"]
# showfiles = ["IXI083-HH-1357-T1_sl63", "IXI079-HH-1388-T1_sl76", "IXI049-HH-1358-T1_sl89", "IXI201-HH-1588-T1_sl71", "IXI083-HH-1357-T1_sl64",
#              "IXI363-HH-2050-T1_sl68", "IXI083-HH-1357-T1_sl61", "IXI211-HH-1568-T1_sl87", "IXI132-HH-1415-T1_sl67", "IXI049-HH-1358-T1_sl85", "IXI612-HH-2688-T1_sl65"]

# MRI: CHAOS
# showfiles = ["CHAOS_MR_SET_21_205928_23000_021_501_T2SPIR_sl20"] #16
# showfiles = ["CHAOS_MR_SET_1_085044_89000_001_801_T1DUAL_InPhase_01_sl15"] #16
# showfiles = ["CHAOS_MR_SET_7_084238_00000_007_601_T1DUAL_InPhase_02_sl23"] #16
# showfiles = ["CHAOS_MR_SET_40_164452_56000_040_401_T2SPIR_sl24"] #8
# showfiles = ["CHAOS_MR_SET_1_085044_89000_001_801_T1DUAL_InPhase_01_sl18"] #8
# showfiles = ["CHAOS_MR_SET_7_084238_00000_007_601_T1DUAL_InPhase_02_sl24"] #8
undersamples = {
    # "sparse4": "Sparse 4",
    # "sparse8": "Sparse 8",
    "sparse16": "Sparse 16",
}
# showfiles = ["CHAOS_MR_SET_1_085044_89000_001_801_T1DUAL_InPhase_01_sl15", "CHAOS_MR_SET_4_090646_76000_004_601_T1DUAL_InPhase_01_sl24",
#              "CHAOS_MR_SET_21_205928_23000_021_501_T2SPIR_sl20", "CHAOS_MR_SET_7_084238_00000_007_601_T1DUAL_InPhase_02_sl23", 
#              "CHAOS_MR_SET_40_164452_56000_040_401_T2SPIR_sl24", "CHAOS_MR_SET_7_083631_73000_007_401_T2SPIR_sl26", 
#              "CHAOS_MR_SET_4_090203_01000_004_401_T2SPIR_sl00", "CHAOS_MR_SET_1_084554_87000_001_501_T2SPIR_sl18", 
#              "CHAOS_MR_SET_18_193414_18000_018_301_T2SPIR_sl08", "CHAOS_MR_SET_7_084238_00000_007_601_T1DUAL_InPhase_02_sl24",
#              "CHAOS_MR_SET_1_085044_89000_001_801_T1DUAL_InPhase_01_sl18", "CHAOS_MR_SET_4_090646_76000_004_601_T1DUAL_InPhase_01_sl20"]

# #CT: Fan
# # showfiles = ["1024"]
# showfiles = ["964", "134", "71", "1258", "1259", "140", "1325", "1363", "121", "1374", "1400", "403", "1397", "515", "10", "522", "1024", "1025", "59"]

# #CT: Parallel
# showfiles = ["189", "392", "383", "384", "59"]

###################################################################################

abs_diff = True
# for CT, this has to be True. For MR, doesn't make any difference. But logically, False
normb4ssim = True
prenorm = False
rot90 = False  # for MR, this has to be True as the NIFTIs are all rotated otherwise. For CT, should be False
ssimfilter = {
    "sparse4": True,
    "sparse8": True,
    "sparse16": True,
}

####################################################################################


# undersamples = {
#     "sparse4": "Sparse 4",
#     "sparse8": "Sparse 8",
#     "sparse16": "Sparse 16",
# }

names = {
    # "bilin_reco": "Interpolated\nSinogram",
    "sparse": "Sparse",
    "reco_unet": "Reconstruction\nUNet",
    # "sino_unet_customup": "Sinogram UNet",
    "sino_unet": "Sinogram UNet",
    "pd_orig": "Primal-Dual",
    "pd_unet": "Primal-Dual\nUNet",
}

for showfile in showfiles:
    imgarr_input = None
    undersamplings = []
    imgarr_output = []
    imgarr_diffs = []
    imgarr_ssimMaps = []
    arr_ssim = []
    for _, undersample in undersamples.items():
        # with open(f'{root_path}/pickles/{showfile}_{undersample}.pickle', 'rb') as handle:
        #     mega_dict = pickle.load(handle)

        # im_inp = mega_dict["fully"]
        im_inp = np.load(pjoin(root_path, 'pd_unet', f'{showfile}_gt.npy')).squeeze()
        # if "underNUFFT" in mega_dict:
        #     im_out = [mega_dict["underNUFFT"]]
        #     methods = ["Undersampled\n(PyNUFFT)"]
        # else:
        im_out = []
        ssimmaps_out = []
        methods = []

        im_out += [np.load(pjoin(root_path, n, f'{showfile}_pred.npy')).squeeze() for n in names]
        methods += [value for _, value in names.items()]

        if prenorm:
            im_inp = min_max(im_inp)
            for i in range(len(im_out)):
                im_out[i] = min_max(im_out[i])

        imgarr_diff = []
        ssimmaps_out = []
        ssimvals = []
        for out in im_out:
            imgarr_diff.append(im_inp - out)
            if normb4ssim:
                ssimval, ssimmap = ssim(
                    im_inp, out, data_range=np.ptp(im_inp), full=True)
            else:
                ssimval, ssimmap = ssim(
                    im_inp, out, data_range=1, full=True)
            ssimvals.append(ssimval)
            ssimmaps_out.append(ssimmap)
        imgarr_output.append(im_out)
        imgarr_ssimMaps.append(ssimmaps_out)
        arr_ssim.append(ssimvals)
        imgarr_diffs.append(imgarr_diff)
        undersamplings.append(undersample)

        if imgarr_input is None:
            imgarr_input = im_inp
            method_names = methods

    imgarr_output = np.array(imgarr_output)
    imgarr_diffs = np.array(imgarr_diffs)
    imgarr_ssimMaps = np.array(imgarr_ssimMaps)
    imgarr_input = np.rot90(imgarr_input) if rot90 else imgarr_input

    nrows = 7
    ncols = 3
    ####################################################################################
    for i, undersampling in enumerate(undersamplings):
        if ssimfilter[list(undersamples.keys())[list(undersamples.values()).index(undersampling)]]:
            pd_orig_ssim = arr_ssim[i][method_names.index(names['pd_orig'])]
            pd_unet_ssim = arr_ssim[i][method_names.index(names['pd_unet'])]
            if pd_orig_ssim > pd_unet_ssim:
                continue

        fig = plt.figure(figsize=(ncols*2, nrows*2))
        axes = []
        ims = []
        # ------
        axes.append(fig.add_subplot(nrows, ncols, round(ncols/2)))
        axes[-1].imshow(imgarr_input, cmap='gray',
                        vmin=-1000, vmax=1000)
        plt.axis(False)
        plt.title('Fully-sampled', fontsize=9)
        # -----
        for j in range(imgarr_output[i].shape[0]):
            out = np.rot90(imgarr_output[i][j]
                           ) if rot90 else imgarr_output[i][j]
            diff = np.rot90(np.abs(imgarr_diffs[i][j]) if abs_diff else imgarr_diffs[i][j]) if rot90 else np.abs(
                imgarr_diffs[i][j]) if abs_diff else imgarr_diffs[i][j]
            ssimmap = np.rot90(
                imgarr_ssimMaps[i][j]) if rot90 else imgarr_ssimMaps[i][j]

            ind = ncols*(j+1)

            axes.append(fig.add_subplot(nrows, ncols, ind+1))
            ims.append(axes[-1].imshow(out, cmap='gray',
                       vmin=-1000, vmax=1000))
            axes[-1].set_ylabel(f'{method_names[j]}', wrap=True, fontsize=12)
            axes[-1].set_title(f'SSIM: {round(arr_ssim[i][j],3)}', fontsize=9)
            axes_clean(axes[-1])

            axes.append(fig.add_subplot(nrows, ncols, ind+2))
            ims.append(axes[-1].imshow(np.abs(diff), cmap='hot',
                       vmin=0. if abs_diff else imgarr_diffs.min(), vmax=1000))
            axes_clean(axes[-1])
            divider = make_axes_locatable(axes[-1])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(ims[-1], cax=cax, orientation='vertical')

            axes.append(fig.add_subplot(nrows, ncols, ind+3))
            ims.append(axes[-1].imshow(ssimmap, cmap='gray',
                       vmin=imgarr_ssimMaps.min(), vmax=imgarr_ssimMaps.max()))
            axes_clean(axes[-1])
            divider = make_axes_locatable(axes[-1])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            plt.colorbar(ims[-1], cax=cax, orientation='vertical')

            if j+1 == imgarr_output[i].shape[0]:  # final subplot
                axes[-1].set_xlabel("SSIM Map", fontsize=12)
                axes[-2].set_xlabel("Difference\nImage", fontsize=12)
                axes[-3].set_xlabel("Reconstruction", fontsize=12)

        # plt.tight_layout()
        # plt.show()
        # plt.show(block=False)
        plt_path = pjoin(root_path, f'{showfile}_{undersampling.replace(" ","_")}')
        plt.savefig(plt_path + ".eps")
        plt.savefig(plt_path + ".png")
        plt.show()
        plt.close()
