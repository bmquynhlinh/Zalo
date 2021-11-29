'''
Author: BUI Mai Quynh Linh
5K
Description:
-- blank database is used for external testing.
'''
import csv
import os
import shutil

import pandas as pd
from matplotlib import pyplot as plt


def remove_blank(train_meta, cols, cols_name):
    data = pd.read_csv(train_meta, sep=',')  # pd.DataFrame.to_numpy(
    data_process = data.iloc[:, [0, 1, cols]]
    data_process = data_process[data_process[cols_name].notna()]  # drop column contains 0
    data_process.to_csv(train_meta.replace('.csv', '_' + cols_name + '.csv'))
    print(data_process)
    return None


def main():
    train_dir = './train/'
    train_meta = train_dir + 'train_meta.csv'
    remove_blank(train_meta, 2, 'mask')
    remove_blank(train_meta, 3, 'distancing')
    remove_blank(train_meta, 4, '5k')

    # data = pd.read_csv('./train/train_mask_enet.csv', sep=',')
    # for idx in range(len(data.axes[0])):
    #     print(data.iloc[idx, 4])
    #     if int(data.iloc[idx, 4]) == int(0.0):
    #         try:
    #             shutil.move(train_dir + 'images_mask/' + data.iloc[idx, 3],train_dir + 'images_mask/0.0/' + data.iloc[idx, 3])
    #         except: continue
    #     else:
    #         try:
    #             shutil.move(train_dir + 'images_mask/' + data.iloc[idx, 3],
    #                     train_dir + 'images_mask/1.0/' + data.iloc[idx, 3])
    #         except: continue


if __name__ == '__main__':
    main()
