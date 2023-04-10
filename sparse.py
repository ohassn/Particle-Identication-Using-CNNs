import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MF

from sklearn.model_selection import cross_val_score
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize

class ParticleRingDataset(Dataset):

    def __init__(self, quantization_size=0.000005):

        self.quantization_size = quantization_size
        self.xy = torch.load('Datasets/pion.pt') + torch.load('Datasets/electron.pt') + torch.load('Datasets/proton.pt') + torch.load('Datasets/kaon.pt')
         
    def __len__(self):
        return len(self.xy)
    
    def shuffle(self):
        np.random.shuffle(self.xy)

    def __getitem__(self, i):

        coords = torch.tensor(self.xy[i][0])
        feats = torch.ones((len(coords), 1))

        # Quantize the input
        discrete_coords, unique_feats = ME.utils.sparse_quantize(
            coordinates=coords,
            features=feats,
            quantization_size=self.quantization_size)

        return discrete_coords, unique_feats, self.xy[i][1]


class CNN(nn.Module):
    def __init__(self, in_channels, num_class, D):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=3,
                stride=2,
                dilation=1,
                bias=True,
                dimension=D),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU())
        self.pool = ME.MinkowskiAvgPooling(2, stride=2, dimension=D)
        self.drop = ME.MinkowskiDropout()
        self.conv2 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                bias=True,
                dimension=D),
            ME.MinkowskiBatchNorm(128),
            ME.MinkowskiReLU())
        self.conv3 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=2,
                bias=True,
                dimension=D),
            ME.MinkowskiBatchNorm(256),
            ME.MinkowskiReLU())
        self.pooling = ME.MinkowskiGlobalMaxPooling()
        self.linear = ME.MinkowskiLinear(256, num_class)

    def forward(self, x):
        out = MF.relu(MF.dropout(self.conv1(x)))
        out = self.pool(out)
        out = MF.relu(MF.dropout(self.conv2(out)))
        out = self.pool(out)
        out = MF.relu(MF.dropout(self.conv3(out)))
        out = self.pool(out)
        out = self.pooling(out)
        return self.linear(out)

# Setting Device
device = torch.device('cuda')

def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

def main(config) -> list:
    # Classification
    net = CNN(
        1,  # in nchannel
        4,  # out_nchannel
        D=2).to(device)

    optimizer = optim.Adam(
        net.parameters(),
        # lr=config['lr'])
        lr=config.lr)

    criterion = nn.CrossEntropyLoss()

    # Dataset, data loader
    data = ParticleRingDataset()
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(data, [int(data.__len__()*0.8), round(data.__len__()*0.1), round(data.__len__()*0.1)])

    print('Train Dataset size: ' + str(train_dataset.__len__()))
    print('Val Dataset size: ' + str(val_dataset.__len__()))
    print('Test Dataset size: ' + str(test_dataset.__len__()))

    train_dataloader = DataLoader(
        train_dataset,
        # batch_size=int(config['batch_size']),
        batch_size=int(config.batch_size),
        collate_fn=ME.utils.batch_sparse_collate,
        shuffle=True)
    
    val_dataloader = DataLoader(
        val_dataset,
        # batch_size=int(config['batch_size']),
        batch_size=int(config.batch_size),
        collate_fn=ME.utils.batch_sparse_collate,
        shuffle=True)
    
    test_dataloader = DataLoader(
        test_dataset,
        # batch_size=int(config['batch_size']),
        batch_size=int(config.batch_size),
        collate_fn=ME.utils.batch_sparse_collate,
        shuffle=True)
    
    loss_train = []
    loss_val = []

    # for epoch in range(config['max_epochs']):
    for epoch in range(config.max_epochs):
        train_loss = []
        val_loss = []

        train_correct, train_samples = 0, 0
        val_correct, val_samples = 0, 0

        train_iter = iter(train_dataloader)
        val_iter = iter(val_dataloader)

        # Training
        net.train()
        for i, data in enumerate(train_iter):

            # forward
            coords, feats, labels = data

            coords = coords.to(device)
            feats = feats.to(device)
            labels = torch.Tensor(labels).to(device)

            out = net(ME.SparseTensor(feats, coords))

            # backward
            optimizer.zero_grad()

            loss = criterion(out.F, labels.long())
            loss.backward()
            optimizer.step()

            train_correct += get_num_correct(out.F, labels.long())
            train_samples += len(out.decomposed_features)

            train_loss.append(loss.item())
        loss_train.append(sum(train_loss)/len(train_loss))

        # Validation
        net.eval()
        for i, data in enumerate(val_iter):

            # forward
            coords, feats, labels = data

            coords = coords.to(device)
            feats = feats.to(device)
            labels = torch.Tensor(labels).to(device)
            
            out = net(ME.SparseTensor(feats, coords))

            loss = criterion(out.F, labels.long())
            loss.backward()

            val_correct += get_num_correct(out.F, labels.long())
            val_samples += len(out.decomposed_features)

            val_loss.append(loss.item())
        loss_val.append(sum(val_loss)/len(val_loss))

        if epoch%10 == 0:
            # print(f'Epoch: {epoch}, Loss: {sum(train_loss)/len(train_loss)}, Val Loss: {sum(val_loss)/len(val_loss)}')
            print(f'Epoch: {epoch}, Loss: {sum(train_loss)/len(train_loss)}, Accuracy: {100* train_correct / train_samples:.2f}')
            print(f'Epoch: {epoch}, Val Loss: {sum(val_loss)/len(val_loss)}, Val Accuracy: {100* val_correct / val_samples:.2f}')

    test_loss = []

    test_correct, test_samples = 0, 0
    test_iter = iter(test_dataloader)

    net.eval()
    for i, data in enumerate(test_iter):

        # forward
        coords, feats, labels = data

        coords = coords.to(device)
        feats = feats.to(device)
        labels = torch.Tensor(labels).to(device)
            
        out = net(ME.SparseTensor(feats, coords))

        loss = criterion(out.F, labels.long())
        loss.backward()

        test_correct += get_num_correct(out.F, labels.long())
        test_samples += len(out.decomposed_features)

        test_loss.append(loss.item())
    print(f'Test Dataset, Loss: {sum(test_loss)/len(test_loss)}, Accuracy: {100* test_correct / test_samples:.2f}')      


# space = [Integer(12, 128, name='batch_size'),
#          Integer(5, 600, name='max_epochs'),
#          Real(10**-5, 10**0, "log-uniform", name='lr')]


# @use_named_args(space)
# def objective(**params):
#     loss = main(params)
#     return -np.mean(loss)


# res_gp = gp_minimize(objective, space, n_calls=100, random_state=0)

# "Best score=%.4f" % res_gp.fun

# print("""Best parameters: 
# - batch_size=%d 
# - max_epoch=%d 
# - learning_rate=%.6f""" % (res_gp.x[0], res_gp.x[1], res_gp.x[2]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--max_epochs', default=600, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)

    config = parser.parse_args()
    main(config)
