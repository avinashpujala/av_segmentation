import pandas as pd
import glob
import os
import downloadMedia


dir_csv = '/home/ubuntu/avinash/code/projects/av_segmentation/data_csv'
dir_out = r'/home/ubuntu/avinash/vol_ebs/audioset'

path_csv = os.path.join(dir_csv, 'balanced_train_segments.csv')


df_as = pd.read_csv(os.path.join(dir_csv, 'balanced_train_segments.csv'), header=None,
                    names=['yt_id', 'start_time', 'stop_time', 'label'])

# downloadMedia.download_av_speech(path_csv, dir_out, n_files=len(df_as))