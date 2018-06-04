#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 18:15:06 2018

@author: Jose Casadiego
"""
 
import numpy as np
import pylab as pl
import time
import scipy
import sklearn
from sklearn import metrics

pl.close("all")
pl.style.use("seaborn-pastel")

def reconstruction(k,M):

    J=np.loadtxt("Data/connectivity.dat")
    Ad=np.copy(J)
    Ad[Ad!=0]=1
    delay=np.loadtxt("Data/delay.dat")
    delay=delay[0]
    
    N=len(Ad)
    spk= [[] for x in xrange(N)]
    ISIs= [[] for x in xrange(N)]
    ex_spk=np.loadtxt("Data/ex_neurons-%d-0.gdf"%(N+1))
    in_spk=np.loadtxt("Data/in_neurons-%d-0.gdf"%(N+2))
    
    start_time = time.time()
    
    for i in range(len(ex_spk)):
    	spk[int(ex_spk[i,0])-1].append(ex_spk[i,1])
    
    for i in range(len(in_spk)):
    	spk[int(in_spk[i,0])-1].append(in_spk[i,1])
    
    for i in range(N):
    	ISIs[i]=(np.array(spk[i][1:])-np.array(spk[i][:-1])).tolist()
    
    i=k
    events= [[] for x in range(len(spk[i])-1)]
    t=[0]*N
    for ti in range(len(spk[i])-1):
    	events[ti]= [[] for x in range(N)]
    	for j in range(N):
    		if j!=i:
    			for t[j] in range(t[j],len(spk[j])):
    				if spk[j][t[j]]+delay-spk[i][ti]>0 and spk[j][t[j]]+delay-spk[i][ti+1]<0:
    					events[ti][j].append(spk[j][t[j]]+delay-spk[i][ti])
    				elif spk[j][t[j]]+delay-spk[i][ti+1]>0:
    					break
                    
    K_events= []
    
    for ti in range(len(spk[i])-1):
    	K_list=[]	
    	for j in range(N):
    		K_list.append(len(events[ti][j]))
    	K_events.append(max(K_list))
    
    
    ISEs=np.zeros((max(K_events)*N,len(K_events)))
    for ti in range(len(K_events)):
    	for j in range(N):
    		a=events[ti][j]
    		k=len(a)
    		for l in range(k):
    			ISEs[l*N+j,ti]=a[l]
    ISIsi=np.asarray(ISIs[i])
    
    d=np.vstack((ISEs,ISIsi))
    D=sklearn.metrics.pairwise.euclidean_distances(d.T)
    
    center=[]
    for ti in range(len(K_events)):
    	center.append(np.mean(D[ti,:]))
    center_index=np.argmin(center)
    
    non_ranked=D[center_index,:]
    closest_index=np.argsort(non_ranked,axis=0)
    
    c=ISEs[:,center_index].reshape(max(K_events)*N,1)
    W=ISEs-c*np.ones((1,len(K_events)))
    
    y=ISIsi.reshape((1,len(K_events)))-ISIsi[center_index]*np.ones((1,len(K_events)))
    X=np.copy(W)
    
    K_vector=np.asarray(K_events)
    K_vector=K_vector[closest_index]
    
    k=K_vector[M]
    
    y=y[0,closest_index]
    y=y.reshape((1,len(K_events)))
    X=X[:,closest_index]
    
    F=sklearn.metrics.pairwise.euclidean_distances(X.T)
    
    print " "
    print "Problem's Characteristics "
    print "==================================="
    print "Unit: %d"%i
    print "Maximum number of spikes during ISI: %d"%max(K_events)
    print "Coefficient of variation: %f"%(np.std(ISIsi)/np.mean(ISIsi))
    print "Skewness of ISIs: %f"%scipy.stats.skew(ISIsi)
    print "Kurtosis of ISIs: %f"%scipy.stats.kurtosis(ISIsi)
    print "Coefficient of variation of X: %f"%scipy.stats.variation(scipy.stats.variation(F))
    print "Maximum number of spikes at center: %d"%K_events[center_index]
    print "Interspike interval at center: %f"%ISIsi[center_index]
    print "Network size: %d"%N
    print "Number of incoming connections: %d"%np.linalg.norm(Ad[i,:],1)
    print "Number of unknowns: %d"%(k*N)
    print "==================================="
    print " "
    
    y=y[:,1:M+1]
    X=X[:k*N,1:M+1]
    
    print "Employing L2 norm optimization"
    g=np.dot(y,np.linalg.pinv(X))
    
    G=np.reshape(g,(k,-1))
    H=G[0,:]
    L=np.copy(H)
    L[L>=0]=0
    fpr, tpr, thresholds = metrics.roc_curve(np.fabs(Ad[i,:]),np.fabs(H),pos_label=1)
    
    print " "
    print "Predictions"
    print "==================================="
    print "AUC score: %f"%metrics.auc(fpr, tpr)
    print "Equations: %d"%M
    print "==================================="
    
    print("--- %s seconds ---" % (time.time() - start_time))
    
    f, axarr = pl.subplots(1,2)
    axarr[0].plot(H.T,"o")
    axarr[0].plot(np.where(Ad[i,:]!=0)[0],H[np.where(Ad[i,:]!=0)[0]].T,"ro",label="True connections")
    axarr[0].set_title('Connectivity',fontsize=15)
    axarr[0].set_xlabel(r'$j$',fontsize=15)
    axarr[0].set_ylabel(r'$\partial h_{%d}/\partial W^{%d}_{j1}$'%(i,i),fontsize=15)
    axarr[0].legend(loc='upper left')
    axarr[1].plot(fpr, tpr, linewidth=3.0)
    axarr[1].set_title('AUC score = %2.4f' % metrics.auc(fpr, tpr),fontsize=15)
    axarr[1].set_xlabel('FPR',fontsize=15)
    axarr[1].set_ylabel('TPR',fontsize=15)
    pl.tight_layout()


neuron=66
ISIs=500
reconstruction(neuron,ISIs)