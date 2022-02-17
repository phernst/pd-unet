import os
from os.path import join as pjoin

import pandas as pd


def single_type(parfan: str, stype: str, undersampling: int, metric: str):
    model_names = [
        "pd_orig",
        "pd_unet_sino",
        "pd_unet",
        "reco_unet",
        "sino_unet",
    ]

    string_dict = {mn: ["", ""] for mn in model_names}

    for model_name in model_names:
        df = pd.read_csv(pjoin(
            "test_out",
            "img_per99_sino_znorm",
            parfan,
            f'{stype}{undersampling}',
            f'Results_{model_name}.csv'))

        string_dict["type"] = f"{parfan}_{stype}{undersampling}"

        mean = df.mean()[f"{metric} (Out)"]
        median = df.median()[f"{metric} (Out)"]
        std = df.std()[f"{metric} (Out)"]

        if metric == 'RMSE':
            mean, median, std = 50000*mean, 50000*median, 50000*std

        string_dict[model_name][0] += f"{mean.round(3)}+-{std.round(3)}"
        string_dict[model_name][1] += str(median)

    print(string_dict)

    with open(pjoin("qualitative", f"PDUNet_consolidated{metric}.txt"),
              "a", encoding='utf-8') as file_obj:
        file_obj.write(string_dict["type"])
        file_obj.write("\n----------------------------\n")

        for mn in model_names:
            file_obj.write(mn)
            file_obj.write(' ')
            file_obj.write(string_dict[mn][0])
            file_obj.write('\n')

        file_obj.write('\nMedian:\n')

        for mn in model_names:
            file_obj.write(mn)
            file_obj.write(' ')
            file_obj.write(string_dict[mn][1])
            file_obj.write('\n')
        file_obj.write('\n')


def create_consolidated(metric: str):
    os.makedirs("qualitative", exist_ok=True)
    if os.path.exists(pjoin("qualitative", f"PDUNet_consolidated{metric}.txt")):
        os.remove(pjoin("qualitative", f"PDUNet_consolidated{metric}.txt"))
    single_type("parallel", "sparse", 4, metric)
    single_type("parallel", "sparse", 8, metric)
    single_type("parallel", "sparse", 16, metric)
    single_type("fan", "sparse", 4, metric)
    single_type("fan", "sparse", 8, metric)
    single_type("fan", "sparse", 16, metric)


if __name__ == '__main__':
    create_consolidated("RMSE")
