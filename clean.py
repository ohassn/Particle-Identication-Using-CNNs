import numpy as np
import torch
import uproot
import awkward as ak
import sys
import matplotlib.pyplot as plt
import torch.nn as nn
import itertools

file = uproot.open("ROOT files/pi+_5GeV_20deg_22deg_1e3.edm4hep.root")['events']
branches = file.arrays()

# gathering all raw data
x_raw = branches['DRICHHits.position.x'].tolist()
y_raw = branches['DRICHHits.position.y'].tolist()

# filtering out empty events
x = [x for x in x_raw if x]
y = [y for y in y_raw if y]

# data array with format [particle type, energy, xy positions, x momentum, y momentum]

fin_x = []
fin_y = []

for i in range(len(x)):

    # removing non circles
    if len(x[i]) < 50 or len(x[i]) > 500:
        continue
    
    # top right
    if min(y[i]) > 600 and max(y[i]) < 1800 and min(x[i]) > 200 and max(x[i]) < 1400: 
        fin_x.extend(x[i])
        fin_y.extend(y[i])

torch.save(fin_x, 'fin_x.pt') 
torch.save(fin_y, 'fin_y.pt') 

