import os
from glob import glob

import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu, wilcoxon


def underDF(root_path, undersample, run_name, under_type):
    df = pd.read_csv(glob(os.path.join(root_path,"**"+undersample+"**"+run_name,"Results_**"+undersample+"**"+run_name+"**.csv"), recursive=True)[0])
    del df["ErrorPercent (Out)"], df["ErrorPercent (Improve)"], df["MSE (Out)"], df["MSE (Improve)"], df["RMSE (Out)"], df["RMSE (Improve)"], df["SSIM (Out)"], df["SSIM (Improve)"]
    df.rename(columns={"ErrorPercent (Under)": "ErrorPercent (Out)", "MSE (Under)": "MSE (Out)", "RMSE (Under)": "RMSE (Out)", "SSIM (Under)": "SSIM (Out)"}, inplace=True)
    df['Model'] = under_type
    return df

root_path = "Output/Final/MRI/CHAOS"
undersample = "sparse16"

names = {
    "reco_unet": "Reconstruction UNet",
    "sino_unet_customup": "Sinogram UNet",
    "pd_orig": "Primal-Dual",
    "pd_unet": "Primal-Dual UNet",
}

dfs = []

#PyNUFFT
df = underDF(root_path, undersample, "reco_unet", "Undersampled (PyNUFFT)")
dfs.append(df)

#Bilinear
df = underDF(root_path, undersample, "sino_unet_customup", "Sinogram Bilinear")
dfs.append(df)

for n in names:
    df = pd.read_csv(glob(os.path.join(root_path,"**"+undersample+"**"+n,"Results_**"+undersample+"**"+n+"**.csv"), recursive=True)[0])
    df['Model'] = names[n]
    dfs.append(df)

df = pd.concat(dfs)
df.to_csv(os.path.join(root_path, f"consolidated_{undersample}.csv"))

###stats
stats = []
y = df[df.Model == names["pd_unet"]]['SSIM (Out)']
for m in df.Model.unique():
    if m == names["pd_unet"]:
        continue

    x = df[df.Model == m]['SSIM (Out)']
    stat = {
        "Model": m,
        "Kolmogorov-Smirnov": ks_2samp(x, y).pvalue,
        "Mann-Whitney": mannwhitneyu(x,y).pvalue,
        "Wilcoxon": wilcoxon(x,y).pvalue
    }
    stats.append(stat)
df = df.from_dict(stats)
df.to_csv(os.path.join(root_path, f"stats_{undersample}.csv"))
