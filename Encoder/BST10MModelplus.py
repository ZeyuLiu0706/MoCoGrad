import pandas as pd
import torch
import pytorch_lightning as pl
import tqdm
import torchmetrics
import math
from urllib.request import urlretrieve
from zipfile import ZipFile
import os
import torch.nn as nn
import numpy as np


# ratings = pd.read_csv(
#     "data/ratings.csv",
#     sep=",",
# )


class PositionalEmbedding(nn.Module):
    """
    Computes positional embedding following "Attention is all you need"
    """

    def __init__(self, max_len, d_model):
        super().__init__()

        # Compute the positional encodings once in log space.
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)



class BST10MModelplus(pl.LightningModule):
    def __init__(self, args=None):
        super().__init__()
        super(BST10MModelplus, self).__init__()
        movies_file = pd.read_csv(
            "/home/admin/LiuZeYu/LibMTL-1.1.6/examples/ML10M/BST10M/data/movies.csv", sep=","
        )

        ratings_file = pd.read_csv(
            "/home/admin/LiuZeYu/LibMTL-1.1.6/examples/ML10M/BST10M/data/ratings.csv",
            sep=",",
        )
        self.save_hyperparameters()
        self.args = args

        self.embeddings_user_id = nn.Embedding(
            int(ratings_file.user_id.max())+1, int(math.sqrt(ratings_file.user_id.max()))+1
        )
        ##Movies
        self.embeddings_movie = nn.Embedding(
            int(movies_file.movie_id.max())+1+20+6, int(math.sqrt(movies_file.movie_id.max()))+1
        )
        # self.positional_embedding = PositionalEmbedding(15, 9)
        self.positional_embedding = PositionalEmbedding(16, 16)
        # Network
        # self.embed1 = nn.Linear(265,72)
        # self.transfomerlayer = nn.TransformerEncoderLayer(72, 3, dropout=0.2)
        self.transfomerlayer = nn.TransformerEncoderLayer(272, 16, dropout=0.1)
        
        self.linear = nn.Sequential(
            nn.Linear(4620,2380),
            nn.LeakyReLU(),
            nn.Linear(2380,1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            # nn.Linear(256, 1),
        )
        
    def encode_input(self,inputs):
        #     features=[user_id,target_id,genre_id,*movie_history,*movie_history_ratings]
        # return torch.tensor(features),torch.tensor(target_movie_rating, dtype=torch.float32)    
        user_id=inputs[:,0:1]

        target_movie_id=inputs[:,1:2]
        genre_id=inputs[:,2:3]     
        movie_history=inputs[:,3:10]
        rating_history=inputs[:,10:]
        #MOVIES
        movie_history = movie_history.squeeze(1)
        rating_history = rating_history.squeeze(1)
        movie_history = self.embeddings_movie(movie_history)
        rating_history = self.embeddings_movie(rating_history)
        target_movie_id = self.embeddings_movie(target_movie_id)
        genre_id = self.embeddings_movie(genre_id)
        transfomer_features = torch.cat((movie_history,rating_history,target_movie_id,genre_id),dim=1)
        #USERS
        user_id = user_id.squeeze(1)
        user_id = self.embeddings_user_id(user_id)
        user_features = user_id
        return transfomer_features, user_features
    

    def forward(self, batch):
        batch=torch.tensor(batch,dtype=int)
        transfomer_features, user_features = self.encode_input(batch)
        positional_embedding = self.positional_embedding(transfomer_features)
        # print("transfomer_features:",transfomer_features.shape)
        # print("positional_embedding",positional_embedding.shape)
        transfomer_features = torch.cat((transfomer_features, positional_embedding), dim=2)
        transformer_output = self.transfomerlayer(transfomer_features)
        transformer_output = torch.flatten(transfomer_features,start_dim=1)
        #Concat with other features
        features = torch.cat((transformer_output,user_features),dim=1)
        output = self.linear(features)
        return output
       
