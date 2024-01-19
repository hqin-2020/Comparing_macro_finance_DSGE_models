import os
import sys
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=200)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
pd.options.display.float_format = '{:.3g}'.format
sns.set(font_scale = 1.0, rc={"grid.linewidth": 1,'grid.color': '#b0b0b0', 'axes.edgecolor': 'black',"lines.linewidth": 3.0}, style = 'whitegrid')
from datetime import datetime
from support import finiteDiff_1D_first, finiteDiff_1D_second
# from shockElasModules import computeElas
from shockElasDecomposition import computeElas
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator
import pickle
import time
now = datetime.now()

current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

import argparse
parser = argparse.ArgumentParser(description="parameter settings")
parser.add_argument("--Delta",type=float,default=1000.)
parser.add_argument("--delta",type=float,default=0.002)
parser.add_argument("--gamma",type=float,default=8.0)
parser.add_argument("--rho",type=float,default=1.00001)
parser.add_argument("--dataname",type=str,default="tests")
parser.add_argument("--zscale",type=float,default=1.0)
parser.add_argument("--q",type=float,default=0.05)
parser.add_argument("--distorted",type=int,default=0)
parser.add_argument("--zmax",type=float,default=2.0)
parser.add_argument("--boundary",type=int,default=0)
parser.add_argument("--logadjustment",type=int,default=1)

args = parser.parse_args()

Delta = args.Delta
delta = args.delta
gamma = args.gamma
rho = args.rho
dataname = args.dataname
zscale = args.zscale
q = args.q
distorted = args.distorted
zmax = args.zmax
boundary = args.boundary
logadjustment = args.logadjustment

filename = "model.npz"

filename_ell = "./output/"+dataname+"/Delta_"+str(Delta)+"_distorted_"+str(distorted)+"/logadjustment_"+str(logadjustment)+"_boundary_"+str(boundary)+"/zmax_"+str(zmax)+"_zscale_"+str(zscale)+"/delta_"+str(delta)+"/q_"+str(q)+"_gamma_"+str(gamma)+"_rho_"+str(rho)+"/"
res = np.load(filename_ell + filename)

zgrid = int(200*zscale+1)
print('grid points for z: ', zgrid)
zscale = 2*zmax/(zgrid-1)
print('grid size for z: ', zscale)


#Load variables from model solution
v = res['V']

phi = res['phi']
beta = res['beta']
eta = res['eta']
a11 = res['a11']
betaz = a11
alpha = res['alpha']
sigma_k = res['sigma_k']
sigma_z = res['sigma_z']
I = res['I']

zbar = 0

adj = res['d'].copy()
logcmk = np.log(res['cons'])
adj[adj<0] = np.min(adj[adj > 0])
logimk = np.log(adj)

#FinDiff

dcdz = finiteDiff_1D_first(logcmk,0,zscale)
ddcddz = finiteDiff_1D_second(logcmk,0,zscale)

dddz = finiteDiff_1D_first(logimk,0,zscale)
dddddz = finiteDiff_1D_second(logimk,0,zscale)

dvdz = finiteDiff_1D_first(v,0,zscale)
ddvddz = finiteDiff_1D_second(v,0,zscale)


if logadjustment==False:
    mu_k = (res['d'] - phi/2*res['d']**2) + beta*res['zz'] - eta*np.ones(res['I']) - np.dot(sigma_k,sigma_k)/2
else:
    mu_k = np.log(1+res['d']*res['phi'])/res['phi'] + beta*res['zz'] - eta*np.ones(res['I']) - np.dot(sigma_k,sigma_k)/2

mu_1 = mu_k 

kdrift = mu_1
kdiffusion = [sigma_k[0]*np.ones(res['I']), sigma_k[1]*np.ones(res['I'])]

zdrift = -betaz*(res['zz']-zbar*np.ones(res['I'])) 
zdiffusion = [sigma_z[0]*np.ones(res['I']), sigma_z[1]*np.ones(res['I'])]

cdrift = dcdz*zdrift + 1/2*ddcddz*(zdiffusion[0]**2+zdiffusion[1]**2) 
cdiffusion = [dcdz*zdiffusion[i] for i in range(len(zdiffusion))]

Cdrift = cdrift + kdrift
Cdiffusion = [cdiffusion[i] + kdiffusion[i] for i in range(len(cdiffusion))]

zerodrift = np.zeros((res['I']))
zerodiffusion = [np.zeros((res['I'])),np.zeros((res['I']))]

Hk = (1-gamma)*res['hk']
Hz = (1-gamma)*res['hz']
H = [Hk, Hz]
sk = res['s1']
sz = res['s2']
S = [sk, sz]
eta = [Hk + sk, Hz + sz]

kdrift = kdrift + (sigma_k[0]*res['s1']+sigma_k[1]*res['s2']) + (1-gamma)*res['hk']*sigma_k[0] + (1-gamma)*res['hz']*sigma_k[1]
zdrift = zdrift + res['mu_z_distorted'] + (1-gamma)*res['hk']*sigma_z[0] + (1-gamma)*res['hz']*sigma_z[1]

sdrift = - delta - Cdrift - 0.5*((Hk + sk)**2 + (Hz + sz)**2)
sdiffusion = [-Cdiffusion[i]+ eta[i] for i in range(len(Cdiffusion))]

ndrift = - 0.5*((Hk + sk)**2 + (Hz + sz)**2)
ndiffusion = [eta[i] for i in range(len(Cdiffusion))]
# print(ndrift)

ambdrift = - 0.5*((sk)**2 + (sz)**2)
ambdiffusion = [S[i] for i in range(len(Cdiffusion))]

misdrift = - 0.5*((Hk)**2 + (Hz)**2)
misdiffusion = [H[i] for i in range(len(Cdiffusion))]

#%%

modelsol = {}
modelsol['stateMatInput'] = [res['zz'][:]]
modelsol['muC'] = Cdrift
modelsol['sigmaC'] = Cdiffusion
modelsol['muc'] = cdrift
modelsol['sigmac'] = cdiffusion
modelsol['muamb'] = ambdrift
modelsol['sigmaamb'] = ambdiffusion
modelsol['mumis'] = misdrift
modelsol['sigmamis'] = misdiffusion
modelsol['muN'] = ndrift
modelsol['sigmaN'] = ndiffusion
modelsol['muzero'] = zerodrift
modelsol['sigmazero'] = zerodiffusion
modelsol['muS'] = sdrift
modelsol['sigmaS'] = sdiffusion
modelsol['muX'] = [zdrift]
modelsol['sigmaX'] = [zdiffusion]
modelsol['nDims'] = 1
modelsol['nShocks'] = 2

#%%

dim = 0

marginals = {}
inverseCDFs = {}
nRange   = list(range(modelsol['nDims']))
axes     = list(filter(lambda x: x != dim,nRange))
condDent = (res['g']*zscale).sum(axis = tuple(axes))
marginals['z'] = condDent.copy()
cumden   = np.cumsum(marginals['z'])
cdf      = interpolate.interp1d(cumden, modelsol['stateMatInput'][dim], fill_value= (modelsol['stateMatInput'][dim][1],  modelsol['stateMatInput'][dim][-2]), bounds_error = False)
inverseCDFs['z'] = cdf

quantile_var = 'Z'
modelsol['quantile_var'] = 'Z'
modelsol['x0'] = np.matrix([[inverseCDFs['z'](0.1)],[inverseCDFs['z'](0.5)],[inverseCDFs['z'](0.9)]])

with open(filename_ell +"model_dis_sol_"+quantile_var, "wb") as f:
    pickle.dump(modelsol, f)

#%%

T = 50*4
dt = 1/4

bc = {'natural':True}

# bc = {}
# bc['a0']  = 0
# bc['first'] = np.matrix([1, 1], 'd')
# bc['level'] = np.matrix([0])
# bc['natural'] = False

muXs = []; 
stateVols = []
SDFVols = [];
NVols = [];
sigmaambVols = [];
sigmamisVols = [];
sigmaCVols = [];
sigmacVols = [];
stateVolsList = []; 
sigmazeroVols = [];
sigmaXs = []
commonInput = {}

# Iterate over state dimensions
for n in range(modelsol['nDims']):
    muXs.append(RegularGridInterpolator(modelsol['stateMatInput'],modelsol['muX'][n]))
    if n == 0:
        commonInput['muC'] = RegularGridInterpolator(modelsol['stateMatInput'], modelsol['muC'])
        commonInput['muc'] = RegularGridInterpolator(modelsol['stateMatInput'], modelsol['muc'])
        commonInput['muS'] = RegularGridInterpolator(modelsol['stateMatInput'], modelsol['muS'])
        commonInput['muamb'] = RegularGridInterpolator(modelsol['stateMatInput'], modelsol['muamb'])
        commonInput['mumis'] = RegularGridInterpolator(modelsol['stateMatInput'], modelsol['mumis'])
        commonInput['muzero'] = RegularGridInterpolator(modelsol['stateMatInput'], modelsol['muzero'])
        commonInput['muN'] = RegularGridInterpolator(modelsol['stateMatInput'], modelsol['muN'])
    # Iterate over shocks dimensions 
    for s in range(modelsol['nShocks']):
        stateVols.append(RegularGridInterpolator(modelsol['stateMatInput'], modelsol['sigmaX'][n][s]))
        if n == 0:
            SDFVols.append(RegularGridInterpolator(modelsol['stateMatInput'], modelsol['sigmaS'][s]))
            NVols.append(RegularGridInterpolator(modelsol['stateMatInput'], modelsol['sigmaN'][s]))
            sigmaambVols.append(RegularGridInterpolator(modelsol['stateMatInput'], modelsol['sigmaamb'][s]))
            sigmamisVols.append(RegularGridInterpolator(modelsol['stateMatInput'], modelsol['sigmamis'][s]))
            sigmaCVols.append(RegularGridInterpolator(modelsol['stateMatInput'], modelsol['sigmaC'][s]))
            sigmacVols.append(RegularGridInterpolator(modelsol['stateMatInput'], modelsol['sigmac'][s]))
            sigmazeroVols.append(RegularGridInterpolator(modelsol['stateMatInput'], modelsol['sigmazero'][s]))

    stateVolsList.append(stateVols)
    def sigmaXfn(n):
        return lambda x: np.transpose([vol(x) for vol in stateVolsList[n] ])
    sigmaXs.append(sigmaXfn(n))
    stateVols = []
    if n == 0:
        commonInput['sigmaS'] = lambda x: np.transpose([vol(x) for vol in SDFVols])
        commonInput['sigmaN'] = lambda x: np.transpose([vol(x) for vol in NVols])
        commonInput['sigmaamb'] = lambda x: np.transpose([vol(x) for vol in sigmaambVols])
        commonInput['sigmamis'] = lambda x: np.transpose([vol(x) for vol in sigmamisVols])
        commonInput['sigmaC'] = lambda x: np.transpose([vol(x) for vol in sigmaCVols])
        commonInput['sigmac'] = lambda x: np.transpose([vol(x) for vol in sigmacVols])
        commonInput['sigmazero'] = lambda x: np.transpose([vol(x) for vol in sigmazeroVols])

commonInput['sigmaX'] = sigmaXs
commonInput['muX']    = lambda x: np.transpose([mu(x) for mu in muXs])
commonInput['T'] = T; commonInput['dt'] = dt;

#%%

###############################################################################################################################################
start_time = time.time()
modelInput = commonInput.copy()
modelInput['sigmaC'] = commonInput['sigmaN']
modelInput['muC']    = commonInput['muN']
print(np.mean(modelInput['muC']([1])))
print(np.mean(modelInput['sigmaC']([1])))
modelInput['sigmaS'] = commonInput['sigmazero']
modelInput['muS']    = commonInput['muzero']

NexpoElas, NpriceElas, NcostElas, _, _ = computeElas(modelsol['stateMatInput'], modelInput, bc, modelsol['x0'])
print("--- %s seconds for the elasticity computation ---" % (time.time() - start_time))
###############################################################################################################################################

start_time = time.time()
modelInput = commonInput.copy()
modelInput['sigmaC'] = commonInput['sigmamis']
modelInput['muC']    = commonInput['mumis']
print(np.mean(modelInput['muC']([1])))
print(np.mean(modelInput['sigmaC']([1])))
modelInput['sigmaS'] = commonInput['sigmazero']
modelInput['muS']    = commonInput['muzero']

misexpoElas, mispriceElas, miscostElas, _, _ = computeElas(modelsol['stateMatInput'], modelInput, bc, modelsol['x0'])
print("--- %s seconds for the elasticity computation ---" % (time.time() - start_time))
###############################################################################################################################################

start_time = time.time()
modelInput = commonInput.copy()
modelInput['sigmaC'] = commonInput['sigmaamb']
modelInput['muC']    = commonInput['muamb']
print(np.mean(modelInput['muC']([1])))
print(np.mean(modelInput['sigmaC']([1])))
modelInput['sigmaS'] = commonInput['sigmazero']
modelInput['muS']    = commonInput['muzero']

ambexpoElas, ambpriceElas, ambcostElas, _, _ = computeElas(modelsol['stateMatInput'], modelInput, bc, modelsol['x0'])
print("--- %s seconds for the elasticity computation ---" % (time.time() - start_time))
###############################################################################################################################################

start_time = time.time()
modelInput = commonInput.copy()
modelInput['sigmaC'] = commonInput['sigmaS']
modelInput['muC']    = commonInput['muS']
print(np.mean(modelInput['muC']([1])))
print(np.mean(modelInput['sigmaC']([1])))
modelInput['sigmaS'] = commonInput['sigmazero']
modelInput['muS']    = commonInput['muzero']

SexpoElas, SpriceElas, ScostElas, _, _ = computeElas(modelsol['stateMatInput'], modelInput, bc, modelsol['x0'])
print("--- %s seconds for the elasticity computation ---" % (time.time() - start_time))

#%%
# res = {'CexpoElas_amb':CexpoElas_amb,\
#        'CpriceElas_amb':CpriceElas_amb,\
#         'CcostElas_amb':CcostEla_amb,\
#         'cexpoElas_amb':cexpoElas_amb,\
#         'cpriceElas_amb':cpriceElas_amb,\
#         'ccostElas_amb':ccostEla_amb,\
#         'CexpoElas_mis':CexpoElas_mis,\
#         'CpriceElas_mis':CpriceElas_mis,\
#         'CcostElas_mis':CcostElas_mis,\
#         'cexpoElas_mis':cexpoElas_mis,\
#         'cpriceElas_mis':cpriceElas_mis,\
#         'ccostElas_mis':ccostElas_mis,\
#         'CexpoElas':CexpoElas,\
#         'CpriceElas':CpriceElas,\
#         'CcostElas':CcostElas,\
#         'cexpoElas':cexpoElas,\
#         'cpriceElas':cpriceElas,\
#         'ccostElas':ccostElas,\
#         'CexpoElas_N':CexpoElas_N,\
#         'CpriceElas_N':CpriceElas_N,\
#         'CcostElas_N':CcostElas_N,\
#         'cexpoElas_N':cexpoElas_N,\
#         'cpriceElas_N':cpriceElas_N,\
#         'ccostElas_N':ccostElas_N,\
res = {'NexpoElas':NexpoElas,\
        'NpriceElas':NpriceElas,\
        'NcostElas':NcostElas,\
        'misexpoElas':misexpoElas,\
        'mispriceElas':mispriceElas,\
        'miscostElas':miscostElas,\
        'ambexpoElas':ambexpoElas,\
        'ambpriceElas':ambpriceElas,\
        'ambcostElas':ambcostElas,\
        'SexpoElas':SexpoElas,\
        'SpriceElas':SpriceElas,\
        'ScostElas':ScostElas}
with open(filename_ell + "model_dis_ela_"+quantile_var, "wb") as f:
    pickle.dump(res, f)

