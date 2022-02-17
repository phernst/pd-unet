import csv
from os.path import join as pjoin

import pandas as pd


def single_type(parfan: str, stype: str, undersampling: int, unet_orig: str,
                precision: int):
    df = pd.read_csv(pjoin(
        "test_out",
        "img_per99_sino_znorm",
        parfan,
        f'{stype}{undersampling}',
        f'Results_pd_{unet_orig}.csv'))

    df = df.rename(columns={'Unnamed: 0': 'Testset Index'})

    median = df.median()["SSIM (Out)"]
    filenames = df[(df["SSIM (Out)"].round(precision) == round(median, precision))][["Testset Index", "Img", "SSIM (Out)"]]

    with open(pjoin("qualitative", f"PD{unet_orig}_r{precision}_filenames.csv"),
              "a", encoding='utf-8') as file_obj:
        csvwriter = csv.writer(file_obj)
        for _, row in filenames.iterrows():
            fullrow = [parfan, stype, f'{undersampling}'] + list(row)
            csvwriter.writerow(fullrow)


def create_median_precision(precision: int):
    unet_orig = 'unet'
    with open(pjoin("qualitative", f"PD{unet_orig}_r{precision}_filenames.csv"),
              "w", encoding='utf-8') as file_obj:
        csvwriter = csv.writer(file_obj)
        csvwriter.writerow(['parfan', 'stype', 'undersampling', 'idx', 'filename', 'ssim'])
    single_type("parallel", "sparse", 4, unet_orig, precision)
    single_type("parallel", "sparse", 8, unet_orig, precision)
    single_type("parallel", "sparse", 16, unet_orig, precision)
    single_type("fan", "sparse", 4, unet_orig, precision)
    single_type("fan", "sparse", 8, unet_orig, precision)
    single_type("fan", "sparse", 16, unet_orig, precision)

    unet_orig = 'orig'
    with open(pjoin("qualitative", f"PD{unet_orig}_r{precision}_filenames.csv"),
              "w", encoding='utf-8') as file_obj:
        csvwriter = csv.writer(file_obj)
        csvwriter.writerow(['parfan', 'stype', 'undersampling', 'idx', 'filename', 'ssim'])
    single_type("parallel", "sparse", 4, unet_orig, precision)
    single_type("parallel", "sparse", 8, unet_orig, precision)
    single_type("parallel", "sparse", 16, unet_orig, precision)
    single_type("fan", "sparse", 4, unet_orig, precision)
    single_type("fan", "sparse", 8, unet_orig, precision)
    single_type("fan", "sparse", 16, unet_orig, precision)


def filter_csv_dataset(filename: str, parfan: str, stype: str, undersampling: str):
    with open(filename, 'r', newline='', encoding='utf-8') as file_obj:
        reader = csv.DictReader(file_obj)
        return [
            row for row in reader
            if row['parfan'] == parfan
            and row['stype'] == stype
            and row['undersampling'] == undersampling
        ]


def find_preferred_slices_with_precision(precision: int):
    unet_path = pjoin("qualitative", f"PDunet_r{precision}_filenames.csv")
    orig_path = pjoin("qualitative", f"PDorig_r{precision}_filenames.csv")

    unet_par4 = filter_csv_dataset(unet_path, 'parallel', 'sparse', '4')
    unet_par8 = filter_csv_dataset(unet_path, 'parallel', 'sparse', '8')
    unet_par16 = filter_csv_dataset(unet_path, 'parallel', 'sparse', '16')
    unet_fan4 = filter_csv_dataset(unet_path, 'fan', 'sparse', '4')
    unet_fan8 = filter_csv_dataset(unet_path, 'fan', 'sparse', '8')
    unet_fan16 = filter_csv_dataset(unet_path, 'fan', 'sparse', '16')

    orig_par4 = filter_csv_dataset(orig_path, 'parallel', 'sparse', '4')
    orig_par8 = filter_csv_dataset(orig_path, 'parallel', 'sparse', '8')
    orig_par16 = filter_csv_dataset(orig_path, 'parallel', 'sparse', '16')
    orig_fan4 = filter_csv_dataset(orig_path, 'fan', 'sparse', '4')
    orig_fan8 = filter_csv_dataset(orig_path, 'fan', 'sparse', '8')
    orig_fan16 = filter_csv_dataset(orig_path, 'fan', 'sparse', '16')

    par4_intersect = [row1 for row1 in unet_par4 for row2 in orig_par4 if row1['idx'] == row2['idx']]
    par8_intersect = [row1 for row1 in unet_par8 for row2 in orig_par8 if row1['idx'] == row2['idx']]
    par16_intersect = [row1 for row1 in unet_par16 for row2 in orig_par16 if row1['idx'] == row2['idx']]
    fan4_intersect = [row1 for row1 in unet_fan4 for row2 in orig_fan4 if row1['idx'] == row2['idx']]
    fan8_intersect = [row1 for row1 in unet_fan8 for row2 in orig_fan8 if row1['idx'] == row2['idx']]
    fan16_intersect = [row1 for row1 in unet_fan16 for row2 in orig_fan16 if row1['idx'] == row2['idx']]

    with open(pjoin("qualitative", f"preferred_r{precision}_filenames.csv"),
              'w', newline='', encoding='utf-8') as file_handle:
        csvwriter = csv.DictWriter(file_handle, unet_par4[0].keys())
        csvwriter.writeheader()
        csvwriter.writerows(par4_intersect)
        csvwriter.writerows(par8_intersect)
        csvwriter.writerows(par16_intersect)
        csvwriter.writerows(fan4_intersect)
        csvwriter.writerows(fan8_intersect)
        csvwriter.writerows(fan16_intersect)

    print('par4:', len(par4_intersect))
    print('par8:', len(par8_intersect))
    print('par16:', len(par16_intersect))
    print('fan4:', len(fan4_intersect))
    print('fan8:', len(fan8_intersect))
    print('fan16:', len(fan16_intersect))


def find_final_preferred_slices():
    with open(pjoin("qualitative", "preferred_r2_filenames.csv"), "r", 
              encoding='utf-8') as file_obj:
        csvreader = csv.DictReader(file_obj)
        preferred_r2 = [
            row for row in csvreader
            if row['undersampling'] == '16'
        ]

    with open(pjoin("qualitative", "preferred_r3_filenames.csv"), "r", 
              encoding='utf-8') as file_obj:
        csvreader = csv.DictReader(file_obj)
        preferred_r3 = [
            row for row in csvreader
            if row['undersampling'] in ['4', '8']
        ]

    with open(pjoin("qualitative", "preferred_slices.csv"), "w", 
              encoding='utf-8') as file_obj:
        csvwriter = csv.DictWriter(file_obj, preferred_r2[0].keys())
        csvwriter.writeheader()
        csvwriter.writerows(preferred_r2)
        csvwriter.writerows(preferred_r3)


if __name__ == '__main__':
    create_median_precision(2)
    find_preferred_slices_with_precision(2)
    create_median_precision(3)
    find_preferred_slices_with_precision(3)
    find_final_preferred_slices()
