import os
from os.path import join as pjoin

import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu, wilcoxon


def under_df(root_path: str, undersample: str, run_name: str, under_type: str):
    df = pd.read_csv(pjoin(
        root_path,
        undersample,
        f"Results_{run_name}.csv"))
    del df["ErrorPercent (Out)"], df["ErrorPercent (Improve)"]
    del df["MSE (Out)"], df["MSE (Improve)"], df["RMSE (Out)"],
    del df["RMSE (Improve)"], df["SSIM (Out)"], df["SSIM (Improve)"]
    df.rename(columns={
            "ErrorPercent (Under)": "ErrorPercent (Out)",
            "MSE (Under)": "MSE (Out)",
            "RMSE (Under)": "RMSE (Out)",
            "SSIM (Under)": "SSIM (Out)"
        }, inplace=True)
    df['Model'] = under_type
    return df


def main(parfan: str, undersample: str):
    root_path = "test_out/img_per99_sino_znorm"
    root_path = pjoin(root_path, parfan)

    names = {
        "reco_unet": "Reconstruction UNet",
        "sino_unet": "Sinogram UNet",
        "pd_orig": "Primal-Dual",
        "pd_unet": "Primal-Dual UNet",
    }

    dfs = []

    # PyNUFFT
    # df = underDF(root_path, undersample, "reco_unet", "Undersampled (PyNUFFT)")
    # dfs.append(df)

    # Bilinear
    df = under_df(root_path, undersample, "sino_unet", "Sinogram Bilinear")
    dfs.append(df)

    for key, name in names.items():
        df = pd.read_csv(pjoin(
            root_path,
            undersample,
            f"Results_{key}.csv"))
        df['Model'] = name
        dfs.append(df)

    df = pd.concat(dfs)
    df.to_csv(pjoin(root_path, f"consolidated_{undersample}.csv"))

    # ## stats
    stats = []
    y = df[df.Model == names["pd_unet"]]['SSIM (Out)']
    for m in df.Model.unique():
        if m == names["pd_unet"]:
            continue

        x = df[df.Model == m]['SSIM (Out)']
        stat = {
            "Model": m,
            "Kolmogorov-Smirnov": ks_2samp(x, y).pvalue,
            "Mann-Whitney": mannwhitneyu(x, y).pvalue,
            "Wilcoxon": wilcoxon(x, y).pvalue
        }
        stats.append(stat)
    df = df.from_dict(stats)
    df.to_csv(os.path.join(root_path, f"stats_{undersample}.csv"))


if __name__ == '__main__':
    main('parallel', 'sparse4')
    main('parallel', 'sparse8')
    main('parallel', 'sparse16')
    main('parallel', 'limited45')
    main('fan', 'sparse4')
    main('fan', 'sparse8')
    main('fan', 'sparse16')
