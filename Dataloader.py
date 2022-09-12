import pandas as pd
import torch
from torch.utils.data import Dataset
import random
import cv2
import os
import numpy as np


class CustomDataLoader(Dataset):
    """
    Args:
        csv_file : str :
        n : int

    Return:
        frames : np. array : (N, H, W, C)
        caption  : np. array : (N, T)
    """
    def __init__(self, csv_file: str, transform=None, n: int = 15, height: int = 224, width: int = 224):
        super(CustomDataLoader, self).__init__()

        self.n = n
        self.height = height
        self.width = width
        self.csv_file = csv_file
        self.transform = transform
        self.main_file = pd.read_csv(csv_file)

    @staticmethod
    def get_frames(path, n, transform, height, width):

        cap = cv2.VideoCapture(path)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_num = random.sample(range(length), n)
        #frames_num = frames_num.sort()
        frame_n = 0
        frames = torch.empty(n, height, width, 3)
        p = 0
        condition = True
        while condition:
            success, frame = cap.read()
            if success:
                if frame_n in frames_num:
                    frame = transform(frame)
                    frame = frame.view(height, width, 3)
                    frames[p, :, :, :] = frame
                    frame_n += 1
                    p += 1
                frame_n += 1
            else:
                condition = False
                cap.release()
        return frames

    def __getitem__(self, item):
        path = self.main_file.iloc[item]['path_to_video']
        caption = self.main_file.iloc[item]['caption']
        n_frames = self.get_frames(path=path, n=self.n, transform=self.transform, height=self.height, width=self.width)
        # if self.transform is not None:
        #     N_frames = self.transform(N_frames)

        return n_frames, caption


    def __len__(self):
        return len(self.main_file)

