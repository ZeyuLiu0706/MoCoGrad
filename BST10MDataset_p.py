import pandas as pd
import torch
import torch.utils.data as data
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
import torch, os
import torchvision.transforms as transforms
import numpy as np
from LibMTL.dataset import AbsDataset

import torch, os
import torchvision.transforms as transforms
import numpy as np
from LibMTL.dataset import AbsDataset

class BST10MDatasetplus(AbsDataset):
        def __init__(self, 
                 path: str,
                 current_task: str,
                 mode: str, 
                 augmentation: bool = False):
            super(BST10MDatasetplus, self).__init__( path=path,
                                             augmentation=augmentation,
                                              current_task=current_task,
                                              mode=mode)
            # self.path=path
            self.test = False

        def _prepare_list(self):
            combine_path=os.path.join(self.path,'{}_{}.csv'.format(self.current_task, self.mode))
            self.ratings_frame = pd.read_csv(
                combine_path,
                delimiter=",",
                # iterator=True,
            )
            return self.ratings_frame


        def __len__(self):
            return len(self.ratings_frame)
        

        def _get_data_labels(self, idx):
            data = self.ratings_frame.iloc[idx]
            user_id = data.user_id
            target_id=data.target_id
            genre_id=data.genre_id
            target_movie_rating = data.target_rating
            movie_history = eval(data.movie_ids)
            movie_history = [int(x) for x in movie_history]
            movie_history_ratings = eval(data.ratings)
            movie_history_ratings = [float(x) for x in movie_history_ratings]
            features=[user_id,target_id,genre_id,*movie_history,*movie_history_ratings]
            features=torch.tensor(features)
            # print("size:",features.shape)
            return features,torch.tensor(target_movie_rating, dtype=torch.float32)           
