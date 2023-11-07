import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import scipy.stats
import sys
from tqdm import tqdm
from plot2d import *
from utils_vamp import *
from args import buildParser
import os
import mdtraj as md
import glob
import pyemma
from deeptime.markov.tools.analysis import mfpt
from pylab import *
plt.rcParams.update({'figure.max_open_warning': 0})
from os import environ
import subprocess
environ['OMP_NUM_THREADS'] = '16'

trajdir = '/home/martinw3/beegfs/martinw3/working/apoe_trajs/'
os.makedirs("stateprobs", exist_ok=True)
os.makedirs("pictures", exist_ok=True)
args = buildParser().parse_args()

thresh = args.threshold
nbrs=args.num_neighbors
nstates=args.num_classes
isoform=args.isoform
tau=args.tau
replicates=args.replicates
dt=args.dt
tau_ns=tau*dt #Convert tau from steps to nanoseconds
scaler = tau * dt / 1000 #Output of MFPT in microseconds
allreps = range(1, replicates + 1)
if args.nter:
    exclude = 'final_'+str(isoform)+'_nter'
else: 
    exclude = 'final_'+str(isoform)
tag = str(nstates)
if args.final:
    tag = 'final_' + tag
    tag2 = tag
if args.nter:
    tag = tag + '_nter'
colors=set_colors(nstates)

with open('exclusions.pkl', 'rb') as f:
    excl = pickle.load(f)
with open('alignedprobs/aligned_probs_{}.pkl'.format(tag), 'rb') as f:
    probs = pickle.load(f)
with open('alignedtrans/aligned_trans_{}.pkl'.format(tag), 'rb') as f:
    trans = pickle.load(f)

exclusions = excl[exclude][str(nstates)]
if type(exclusions) == int:
    exclusions=[exclusions]
selections = [x for x in allreps if x not in exclusions]
embd = []
all_probs = []
transitions = []
for i in range(1, replicates+1):
    if i in exclusions:
        print("Model {} excluded from analysis due to failed ITS".format(i))
    else:
        if args.nter:
            embd_tmp = np.concatenate(np.load('data/{}_{}_nter/embeddings_{}.npz'.format(isoform, i, tag2))['arr_0'])
        else:
            embd_tmp = np.concatenate(np.load('data/{}_{}/embeddings_{}.npz'.format(isoform, i, tag2))['arr_0'])
        embd.append(embd_tmp)
        all_probs.append(probs[isoform][i-1])
        transitions.append(trans[isoform][i-1])
avg_embd = np.mean(embd, axis=0)

all_trajs = []
for i in range(len(all_probs)):
    trajs = []
    for j in range(len(all_probs[i])): 
        traj = []
        for k in range(len(all_probs[i][j])):  
            traj.append(all_probs[i][j][k].argmax())
        traj = np.array(traj)
        trajs.append(traj)
    all_trajs.append(trajs)
        
state_prob=[]
for i in range(len(all_trajs)):
    state_prob.append(np.zeros(nstates))
        
for i in range(len(all_trajs)):
    for j in range(len(all_trajs[i])):
        for k in range(nstates):
            state_prob[i][k] = state_prob[i][k] + sum(all_trajs[i][j]==k)/(len(all_trajs[i][j])*len(all_trajs[i]))

state_probs = np.array(state_prob)*100
np.savetxt('stateprobs/{}_{}.txt'.format(isoform, tag), state_probs)

probs_v = []
for i in range(len(all_probs)):
    probs_v.append(np.vstack(all_probs[i]))
probs_v_avg = np.mean(probs_v, axis=0)
def StateCount(lst, state):
    count = 0
    for val in lst:
        if val == state:
            count+=1
    return count
np.save("avgprobs/avg_probs_{}_{}_states.npy".format(isoform, tag), probs_v_avg)

tmp = []
for i in range(len(probs_v_avg)):
    tmp.append(np.argmax(probs_v_avg[i]))
avg_prob=np.zeros(nstates)
for i in range(len(tmp)):
    for k in range(nstates):
        avg_prob[k] += sum(tmp[i]==k)

thresholds = [0.50, 0.60, 0.75, 0.85, 0.90, 0.95] #In case you want to run for multiple thresholds
for t in tqdm(thresholds, desc="Generating state maps"):
    y_m_t = []
    traj_inds = []
    tmp = []
    tmp_ind = []
    for j in range(len(probs_v_avg)):
        if probs_v_avg[j].max()>t:
            tmp.append(np.argmax(probs_v_avg[j]))
            tmp_ind.append(j)
    tmp = np.array(tmp)
    tmp_ind = np.array(tmp_ind)
    y_m_t.append(tmp)
    traj_inds.append(tmp_ind)
    print("Threshold: {}".format(t))
    print("Frames Used: {}".format(len(y_m_t[0])))
    for i in range(nstates):
        cnt = StateCount(y_m_t[0], i)
        poss = avg_prob[i]
        percnt = cnt/poss*100
        print("State {} contains {} of {} possible frames ({}%)".format(i, cnt, poss, percnt))
    if len(y_m_t[0]) != 0:
        rc('axes', linewidth=2)
        f, ax = plt.subplots(1,1, figsize=(8,5))
        plot_state_map(avg_embd[traj_inds[0],0], avg_embd[traj_inds[0],1], y_m_t[0], cmap=colors, ax=ax, ncontours=100)
        fontsize = 10
        for tick in ax.xaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
            tick.label1.set_fontweight('bold')
        for tick in ax.yaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
            tick.label1.set_fontweight('bold')
        plot_state_map(avg_embd[:,0], avg_embd[:,1], np.zeros(avg_embd[:,0].shape[0]), ax=ax, alpha=0.1, mask=True, cbar=False,cmap='Greys')
        plt.savefig('pictures/states_{}_{}_{}.svg'.format(isoform, t, tag))
        plt.clf()

rc('axes', linewidth=2)
f, ax = plt.subplots(1,1, figsize=(8,5), dpi=100)
plot_free_energy(avg_embd[:,0], avg_embd[:,1],ax=ax)
fontsize = 10
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
    tick.label1.set_fontweight('bold')
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
    tick.label1.set_fontweight('bold')
plt.savefig('pictures/fel_{}_{}.svg'.format(isoform, tag))
plt.clf()

#Generation of implied timescales plots for all replicates 
#as well as a combined plot with confidence interval 
max_tau = int(ceil(len(all_probs[0][0])/2))
units='ns'
lags = np.arange(10, max_tau, 10)
its = []
for i in tqdm(range(len(all_probs)), desc="Generating Implied Timescales"): 
    its.append(get_its(all_probs[i], lags))
    if args.nter:
        plot_its(its[i], lags, ylog=False, modifier=tag, save_folder="data/{}_{}_nter".format(isoform, selections[i]))
    else:
        plot_its(its[i], lags, ylog=False, modifier=tag, save_folder="data/{}_{}".format(isoform, selections[i]))
    plt.clf()
its = np.array(its)

mean_its = np.zeros((nstates-1,len(lags)))
up_its = np.zeros((nstates-1,len(lags)))
down_its = np.zeros((nstates-1,len(lags)))

for i in range(nstates-1):
    for j in range(len(lags)):
        mean_its[i,j], up_its[i,j], down_its[i,j] = mean_confidence_interval(its[:,i,j])

rc('axes', linewidth=2)
f, ax = plt.subplots(1,1, figsize=(8,5), dpi=200)
for index in range(nstates-1):
    ax.semilogy(lags*dt, mean_its[index]*dt)
    ax.fill_between(lags*dt, down_its[index]*dt, up_its[index]*dt, alpha=0.5)
ax.fill_between(lags*dt, lags*dt, alpha=0.5, color='grey')
xmax=np.max(lags)
ax.semilogy(lags*dt, lags*dt, color='black')
ax.set_xlim([1.0*dt, xmax*dt])
ax.set_xlabel('lagtime / %s' % units, fontweight='bold', fontsize=15)
ax.set_ylabel('timescale / %s' % units, fontweight='bold', fontsize=15)
ax.axvline(tau_ns, color='black')
fontsize = 13
ax = gca()
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
    tick.label1.set_fontweight('bold')
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
    tick.label1.set_fontweight('bold')
plt.savefig('pictures/its_{}_{}.svg'.format(isoform, tag))

if not args.final:  #remaining analysis only done if DeepMSM has been retrained
    sys.exit()

mfpts = []
pathin = []
pathout = []
for matrix in transitions:
    mfpt_tmp = np.zeros((nstates, nstates))
    in_tmp = np.zeros(nstates)
    out_tmp = np.zeros(nstates)
    for i in range(nstates):
        others = list(range(nstates))
        others.remove(i)
        in_tmp[i] = mfpt(matrix, target=i, origin=others, tau=scaler)
        out_tmp[i] = mfpt(matrix, target=others, origin=i, tau=scaler)
        for j in range(nstates):
            mfpt_tmp[i, j] = mfpt(matrix, target=i, origin=j, tau=scaler)
    mfpts.append(mfpt_tmp)
    pathin.append(in_tmp)
    pathout.append(out_tmp)
    
mfpts_mean = np.around(np.mean(mfpts, axis=0), decimals=2)
in_mean = np.around(np.mean(pathin, axis=0), decimals=2)
out_mean = np.around(np.mean(pathout, axis=0), decimals=2)
mfpts_std = np.around(np.std(mfpts, axis=0), decimals=2)
in_std = np.around(np.std(pathin, axis=0), decimals=2)
out_std = np.around(np.std(pathout, axis=0), decimals=2)

labs = []
in_labs = []
out_labs = []
for i in range(nstates):
    in_labs.append(str(str(in_mean[i]) + ' \u00b1 ' + str(in_std[i]) + ' \u03bcs'))
    out_labs.append(str(str(out_mean[i]) + ' \u00b1 ' + str(out_std[i]) + ' \u03bcs'))
    labs_tmp = []
    for j in range(nstates):
        labs_tmp.append(str(str(mfpts_mean[i][j]) + ' \u00b1 ' + str(mfpts_std[i][j]) + ' \u03bcs'))
    labs.append(labs_tmp)
os.makedirs("mfpts", exist_ok=True)

np.savetxt('mfpts/labs_{}_{}.csv'.format(isoform, tag), labs, fmt='%s', delimiter=",")
np.savetxt('mfpts/trans_{}_{}.csv'.format(isoform, tag), np.average(transitions, axis=0), fmt='%s', delimiter=",")
np.savetxt('mfpts/in_{}_{}.csv'.format(isoform, tag), in_labs, fmt='%s', delimiter=",")
np.savetxt('mfpts/out_{}_{}.csv'.format(isoform, tag), out_labs, fmt='%s', delimiter=",")

steps = int(floor(len(all_probs[0][0])/tau))
preds = []
ests = []
for i in tqdm(range(len(all_probs)), desc="Generating CK-Tests"):
    pred, est = get_ck_test(all_probs[i], steps=steps, tau=tau)
    preds.append(pred)
    ests.append(est)
preds = np.array(preds)
ests = np.array(ests)

mean_pred = np.zeros((nstates,nstates,steps))
up_pred = np.zeros((nstates,nstates,steps))
down_pred = np.zeros((nstates,nstates,steps))
mean_est = np.zeros((nstates,nstates,steps))
up_est = np.zeros((nstates,nstates,steps))
down_est = np.zeros((nstates,nstates,steps))

for i in range(nstates):
     for j in range(nstates):
         for k in range(steps):
             mean_pred[i,j,k], up_pred[i,j,k], down_pred[i,j,k] = mean_confidence_interval(preds[:,i,j,k], confidence=0.95)
             mean_est[i,j,k], up_est[i,j,k], down_est[i,j,k] = mean_confidence_interval(ests[:,i,j,k], confidence=0.95)

pred = mean_pred
est = mean_est
 
fontsize = 14
fig, ax = plt.subplots(nstates, nstates, sharex=True, sharey=True, figsize=(10,8),dpi=300)
for index_i in range(nstates):
    for index_j in range(nstates):
 
        ax[index_i][index_j].plot(range(0, int(steps*tau*dt), int(tau*dt)), est[index_i, index_j], color='b', linestyle='--')
        ax[index_i][index_j].fill_between(range(0, int(steps*tau*dt), int(tau*dt)), down_pred[index_i, index_j], up_pred[index_i, index_j], alpha=0.4, color='blue')
        ax[index_i][index_j].set_title(str(index_i+1)+'->'+str(index_j+1), fontweight='bold', fontsize=15)
        ax[index_i][index_j].errorbar(range(0, int(steps*tau*dt), int(tau*dt)), est[index_i, index_j], est[index_i,index_j]-down_est[index_i, index_j], color='red')
 
        for tick in ax[index_i][index_j].xaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
            tick.label1.set_fontweight('bold')
        for tick in ax[index_i][index_j].yaxis.get_major_ticks():
            tick.label1.set_fontsize(fontsize)
            tick.label1.set_fontweight('bold')
ax[0][0].set_ylim((-0.1, 1.1))
ax[0][0].set_xlim((0, steps*tau/10))
ax[0][0].axes.get_xaxis().set_ticks(np.round(np.linspace(0, steps*tau/10, 5)))
plt.tight_layout()
fig.legend([pred[0][0], est[0][0]], ['Estimates', 'Predicted'], loc='upper center', ncol=2, bbox_to_anchor=(0.5, -0.1))
ax[nstates-1, nstates//2].set_xlabel('time [ns]', fontweight='bold')
ax[nstates//2, 0].set_ylabel('Probability', fontweight='bold')
 

plt.savefig('pictures/ck_{}_{}.svg'.format(isoform, tag))

#Average attention should be calculated beforehand
if args.nter:
    avg_attn=np.load('attns/average_attn_{}_{}_nter.npz'.format(isoform, nstates))['arr_0']
else:
    avg_attn=np.load('attns/average_attn_{}_{}.npz'.format(isoform, nstates))['arr_0']

attn=np.concatenate(avg_attn)

#load neighbor indices and trajectories
data = []
#iterable for loading all data
isoforms = ["2", "3", "4", "61", "c", "j"]

#files are named sequentially, modifiers added to use the correct numbers
mod = {"2":0, "3":600, "4":1200, "61":1800, "c":2400, "j":3000}

if args.isoform != "all":
    if args.nter:
        directory = trajdir + '{}/nter'.format(args.isoform)
    else:
        directory = trajdir + '{}'.format(args.isoform)
    indlist=sorted(glob.glob(os.path.join(directory, 'ind*.npz')))
    print("First file:", os.path.join(directory, "ind_{}nbrs_{}ns_{:04d}.npz".format(nbrs, dt, int(mod[args.isoform])+1)))
    for i in tqdm(range(len(indlist)), desc="Loading neighbor indices and trajectories"):
        ind = np.load(os.path.join(directory,"ind_{}nbrs_{}ns_{:04d}.npz".format(nbrs, dt, i+int(mod[args.isoform])+1)))['arr_0']
        data.append(ind)
else:
    for isoform in isoforms:
        if args.nter:
            directory = trajdir + '{}/nter'.format(isoform)
        else:
            directory = trajdir + '{}'.format(isoform)
        indlist=sorted(glob.glob(os.path.join(directory, 'ind*.npz')))
        print("Loading isoform {}. First file:".format(isoform), os.path.join(directory, "ind_{}nbrs_{}ns_{:04d}.npz".format(nbrs, dt, int(mod[isoform])+1)))
        for i in tqdm(range(len(indlist)), desc="Loading neighbor indices and trajectories"):
            ind = np.load(os.path.join(directory,"ind_{}nbrs_{}ns_{:04d}.npz".format(nbrs, dt, i+int(mod[isoform])+1)))['arr_0']
            data.append(ind)

inds=np.concatenate(data)

statemax = []
tmax = []
tmax_ind = []
for j in range(len(probs_v_avg)):
    statemax.append(np.argmax(probs_v_avg[j]))
    if probs_v_avg[j].max()>thresh:
        tmax.append(np.argmax(probs_v_avg[j]))
        tmax_ind.append(j)
tmax = np.array(tmax)
tmax_ind = np.array(tmax_ind)

score = np.zeros((nstates, inds.shape[1], inds.shape[1]))
adjscore = np.zeros((nstates, inds.shape[1], inds.shape[1]))
adjtmp = np.zeros((inds.shape[1], inds.shape[1]))
for i in tqdm(range(inds.shape[0]), desc="Getting attention scores"):
    adjtmp = np.zeros((inds.shape[1], inds.shape[1]))
    adjtmp[np.arange(adjtmp.shape[0])[:,None],inds[i]] = attn[i]
    score[statemax[i]] += adjtmp
for i in tqdm(range(len(tmax)), desc="Getting threshold attention scores"):
    adjtmp = np.zeros((inds.shape[1], inds.shape[1]))
    adjtmp[np.arange(adjtmp.shape[0])[:,None],inds[tmax_ind[i]]] = attn[tmax_ind[i]]
    adjscore[tmax[i]] += adjtmp

for i in range(nstates):
    cnt = StateCount(tmax, i)
    poss = avg_prob[i]
    score[i] /= poss
    adjscore[i] /= cnt

np.save('attns/score_{}_{}.npy'.format(isoform, tag), score)
np.save('attns/adjscore_{}_{}.npy'.format(isoform, tag), adjscore)

pdb = open(os.path.join(directory, "ca.pdb"), "r") #file with only alpha carbons
pdbdata=[]
for line in pdb:
    tmpline=line.split()
    if tmpline[0] == 'ATOM':
        pdbdata.append(tmpline)

os.makedirs("attn_plots", exist_ok=True)

for i in range(nstates):
    plt.set_cmap('Reds')
    fig, axes = plt.subplots(figsize=(30,30),dpi=200)
    axes.imshow(score[i])
    axes.hlines(np.arange(0,len(score[i]))-0.5,-0.5,len(score[i])-0.5,color='k',linewidth=0.5,alpha=0.5)
    axes.vlines(np.arange(0,len(score[i]))-0.5,-0.5,len(score[i])-0.5,color='k',linewidth=0.5,alpha=0.5)
    axes.set_xticks(np.arange(0,len(score[i]),5))
    axes.set_xticklabels(np.arange(int(pdbdata[0][4]),int(pdbdata[-1][4]),5))
    axes.set_yticks(np.arange(0,len(score[i]),5))
    axes.set_yticklabels(np.arange(int(pdbdata[0][4]),int(pdbdata[-1][4]),5))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('attn_plots/{}_state{}_{}.svg'.format(isoform, i, tag))
    plt.clf()
for i in range(nstates):
    plt.set_cmap('Reds')
    fig, axes = plt.subplots(figsize=(30,30),dpi=200)
    axes.imshow(adjscore[i])
    axes.hlines(np.arange(0,len(score[i]))-0.5,-0.5,len(score[i])-0.5,color='k',linewidth=0.5,alpha=0.5)
    axes.vlines(np.arange(0,len(score[i]))-0.5,-0.5,len(score[i])-0.5,color='k',linewidth=0.5,alpha=0.5)
    axes.set_xticks(np.arange(0,len(score[i]),5))
    axes.set_xticklabels(np.arange(int(pdbdata[0][4]),int(pdbdata[-1][4]),5))
    axes.set_yticks(np.arange(0,len(score[i]),5))
    axes.set_yticklabels(np.arange(int(pdbdata[0][4]),int(pdbdata[-1][4]),5))
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('attn_plots/{}_adj_state{}_{}.svg'.format(isoform, i, tag))
    plt.clf()

scales=[]
for i in range(len(score)):
    scales.append(scale(score[i].sum(axis=0)))
scores=np.stack(scales)

# Protein is too big to plot in a single line, so it is chunked
outlen=50
chunks=int(ceil(len(score[0])/outlen)) 

for i in range(chunks):
    ind1 = i * outlen
    ind2= (i + 1) * outlen
    if ind2 > len(score[0]):
        ind2 = len(score[0])
    residues=[]
    for j in range(ind1, ind2):
        residues.append(pdbdata[j][3]+pdbdata[j][4])
    fig, ax = plt.subplots(figsize=(10,2),dpi=200)
    ax.imshow(scores[:,ind1:ind2])
    ax.set_xticks(np.arange(0,ind2-ind1))
    ax.set_xticklabels(residues, rotation=90)
    ax.set_yticks(np.arange(nstates))
    ax.set_yticklabels(np.arange(nstates))
    ax.hlines(np.arange(0,nstates)-0.5,-0.5,ind2-ind1-0.5,color='k',linewidth=1,alpha=0.5)
    ax.vlines(np.arange(0,ind2-ind1+1)-0.5,-0.5,nstates-0.5,color='k',linewidth=0.5,alpha=0.5)
    #cbar = plt.colorbar(h, ax=ax)
    #for t in cbar.ax.get_yticklabels():
    #     t.set_fontsize(25)

    fontsize = 10
    ax = gca()

    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontweight('bold')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('attn_plots/combscore_{}_{}_{}_{}.svg'.format(isoform, nstates, ind1+1, ind2))
    plt.clf()

scales=[]
for i in range(len(score)):
    scales.append(scale(adjscore[i].sum(axis=0)))
scores=np.stack(scales)

# Protein is too big to plot in a single line, so it is chunked
outlen=50
chunks=int(ceil(len(score[0])/outlen)) 

for i in range(chunks):
    ind1 = i * outlen
    ind2= (i + 1) * outlen
    if ind2 > len(score[0]):
        ind2 = len(score[0])
    residues=[]
    for j in range(ind1, ind2):
        residues.append(pdbdata[j][3]+pdbdata[j][4])
    fig, ax = plt.subplots(figsize=(10,2),dpi=200)
    ax.imshow(scores[:,ind1:ind2])
    ax.set_xticks(np.arange(0,ind2-ind1))
    ax.set_xticklabels(residues, rotation=90)
    ax.set_yticks(np.arange(nstates))
    ax.set_yticklabels(np.arange(nstates))
    ax.hlines(np.arange(0,nstates)-0.5,-0.5,ind2-ind1-0.5,color='k',linewidth=1,alpha=0.5)
    ax.vlines(np.arange(0,ind2-ind1+1)-0.5,-0.5,nstates-0.5,color='k',linewidth=0.5,alpha=0.5)
    #cbar = plt.colorbar(h, ax=ax)
    #for t in cbar.ax.get_yticklabels():
    #     t.set_fontsize(25)

    fontsize = 10
    ax = gca()

    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontweight('bold')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('attn_plots/combadjscore_{}_{}_{}_{}.svg'.format(isoform, nstates, ind1+1, ind2))
    plt.clf()

probs_avg = np.mean(all_probs, axis=0)
traj_state = []
traj_ind = []
for i in tqdm(range(len(probs_avg)), desc="Getting trajectory indices over threshold"):
    tmpstate = []
    tmpind = []
    for j in range(len(probs_avg[i])):
        if probs_avg[i][j].max() > thresh:
            tmpstate.append(np.argmax(probs_avg[i][j]))
            tmpind.append(j)
    traj_state.append(tmpstate)
    traj_ind.append(tmpind)
'''
os.makedirs("state_trajectories", exist_ok=True)
top = os.path.join(directory, "structure.pdb")
for i in tqdm(range(len(traj_state)), desc="Splitting trajectories"):
    trajout = [ [] for _ in range(nstates) ]
    for j in range(len(traj_state[i])):
        trajout[traj_state[i][j]].append(traj_ind[i][j]+1)
    traj_in = os.path.join(directory, '{:04d}.xtc'.format(i+int(mod[isoform])+1))
    for n in range(nstates):
        if len(trajout[n]) != 0:
            with open("state_{}_{}.ndx".format(isoform, n), "ab") as f:
                f.write(b"[ frames ]\n")
                np.savetxt(f, np.transpose(trajout[n]).astype(int), fmt='%i', delimiter=" ")
                f.close()
        out_traj = 'state_trajectories/{}_{}_cluster_{}_{}.xtc'.format(isoform, tag, n, i+1)
        index = 'state_{}_{}.ndx'.format(isoform, n)
        if os.path.isfile(index):
            p = subprocess.Popen(['gmx', 'trjconv', '-f', traj_in, '-s', top, '-fr', index, '-o', out_traj], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
            p.communicate(b'0\n')
            p.wait()
            os.remove(index)
'''
