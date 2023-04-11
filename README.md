# Particle-Identication-Using-CNNs

Purpose of this model is to classify 4 particle types (pions, electrons, protons, kaons) using sparse CNNs. Minkowski Engine is used for sparse convolutions.

## Dependencies:

a) Minkowski Engine 

b) scikit-learn (optional).

## 1) Data Generation

You will require eic-shell (https://eic.phy.anl.gov/tutorials/eic_tutorial/getting-started/quickstart/).

Enter eic-shell:

```./eic-shell```

Enter nightly:

```source /opt/detector/setup.sh```

Generate data, as an example:

```
npsim --compact epic.xml --enableGun --gun.particle "e-" --gun.energy "5*GeV" --gun.thetaMin "3*deg" 
--gun.thetaMax "50*deg" --gun.phiMin "50*deg" --gun.phiMax "85*deg" --gun.distribution "cos(theta)" 
--numberOfEvents 100000 --outputFile e-_5GeV_20deg_22deg_1e5.edm4hep.root
```

## 2) Data filtration

```clean.py``` is used to filter data in an appropriate format for Minkowski Engine. You will need to write your own cleaning code to use the ```dense.py``` code.

## 3) Training 

```sparse.py``` is used, support for Bayesian hyperparameter optimization is provided in the code using scikit-learn. 

## Citations

- [4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks, CVPR'19](https://arxiv.org/abs/1904.08755), [[pdf]](https://arxiv.org/pdf/1904.08755.pdf)
