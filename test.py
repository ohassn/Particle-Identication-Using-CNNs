import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import MinkowskiEngine as ME


class Dataset(Dataset):

    fin_x = torch.load('Datasets/fin_x.pt')
    def __init__(
        self,
        dataset_size=len(fin_x),
        quantization_size=0.005):

        self.dataset_size = dataset_size
        self.quantization_size = quantization_size

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, i):

        # pion
        fin_x = torch.load('Datasets/fin_x.pt')
        fin_y = torch.load('Datasets/fin_y.pt')

        coords = np.transpose((fin_x, fin_y))
        feats = torch.ones((len(fin_x),1))
        labels = torch.ones(len(fin_x))*0

        # print(coords)

        # Quantize the input
        discrete_coords, unique_feats, unique_labels = ME.utils.sparse_quantize(
            coordinates=coords,
            features=feats,
            labels=labels,
            quantization_size=self.quantization_size,
            ignore_label=-100)
        
        # print(discrete_coords)
        return discrete_coords, unique_feats, unique_labels

class CNN(nn.Module):
    def __init__(self, in_channels, num_class, D):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=in_channels,
                out_channels=8,
                kernel_size=3,
                stride=1,
                dilation=1,
                bias=True,
                dimension=D),
            ME.MinkowskiBatchNorm(8),
            ME.MinkowskiReLU())
        self.pool = ME.MinkowskiMaxPooling(2, stride=2, dimension = D)
        self.conv2 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                stride=1,
                dilation=1,
                bias=True,
                dimension=D),
            ME.MinkowskiBatchNorm(16),
            ME.MinkowskiReLU())
        self.fc1 = ME.MinkowskiLinear(16, num_class)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.fc1(x)

        return x

def collation_fn(data_labels):
    coords, feats, labels = list(zip(*data_labels))
    coords_batch, feats_batch, labels_batch = [], [], []

    # Generate batched coordinates
    coords_batch = ME.utils.batched_coordinates(coords)

    # Concatenate all lists
    feats_batch = torch.from_numpy(np.concatenate(feats, 0)).float()
    labels_batch = torch.from_numpy(np.concatenate(labels, 0))

    return coords_batch, feats_batch, labels_batch

def main(config):
    # Binary classification
    net = CNN(
        1,  # in nchannel
        2,  # out_nchannel
        D=2)

    optimizer = optim.Adam(
        net.parameters(), 
        lr=config.lr)

    criterion = nn.CrossEntropyLoss()

    # Dataset, data loader
    train_dataset = Dataset()

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        collate_fn=ME.utils.batch_sparse_collate,
        num_workers=1)

    accum_loss, accum_iter, tot_iter = 0, 0, 0

    for epoch in range(config.max_epochs):
        train_iter = iter(train_dataloader)
        # Training
        net.train()
        for i, data in enumerate(train_iter):

            # forward
            coords, feats, labels = data
            out = net(ME.SparseTensor(feats.float(), coords))

            # backward
            optimizer.zero_grad()
            loss = criterion(out.F.squeeze(), labels.long())
            loss.backward()
            optimizer.step()

            accum_loss += loss.item()
            accum_iter += 1
            tot_iter += 1
    
            if tot_iter % 10 == 0 or tot_iter == 1:
                print(
                    f'Epoch: {epoch} iter: {tot_iter}, Loss: {accum_loss / accum_iter}'
                )
                accum_loss, accum_iter = 0, 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--max_epochs', default=10, type=int)
    parser.add_argument('--lr', default=0.1, type=float)

    config = parser.parse_args()
    main(config)