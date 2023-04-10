import numpy as np
import torch
import uproot

file = uproot.open("ROOT files/pi+_5GeV_20deg_22deg_1e4.edm4hep.root")['events']
branches = file.arrays()

# gathering all raw data
x_raw = branches['DRICHHits.position.x'].tolist()
y_raw = branches['DRICHHits.position.y'].tolist()

# filtering out empty events
x = [x for x in x_raw if x]
y = [y for y in y_raw if y]

xy = []

for i in range(len(x)):
    # removing non circles
    if len(x[i]) < 50 or len(x[i]) > 500:
        continue

    # coordinates and target
    xy.append((np.stack((x[i], y[i]), axis=-1),0))

    # top right
    # if min(y[i]) > 600 and max(y[i]) < 1900 and min(x[i]) > 200 and max(x[i]) < 1500: 
    #     xy.append((np.stack((x[i], y[i]), axis=-1),0))

print(len(xy))
torch.save(xy, 'pion.pt') 
