#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 18:15:06 2018

reconstruction.py reconstruct the synaptic connectivity of spiking neural 
networks from spike trains alone.

Output
----------
Figure: it is composed of two panels, (i) shows the inferred proxies for
connectivity where true connections are highlighted in red, and (ii) shows the
Receiver-Operating-Characteristics curve for the predicted links.

Accompanying material to "Inferring network connectivity from event timing
patterns".

@author: Jose Casadiego
"""
 
import numpy as np
import pylab as pl
import time
import scipy
import sklearn
from sklearn import metrics

def reconstruction(neuron,M):
    """
    reconstruction(neuron,M) reconstructs the incoming synaptic links for a
    selected neuron from recorded spike trains alone.
    
    Parameters:
    ------------------
    neuron: postsynaptic neuron ID whose incoming links we want to recover.
    M:      number of events employed for reconstruction.
            
    Output
    ----------
    Figure: it is composed of two panels, (i) shows the inferred proxies for
    connectivity where true connections are highlighted in red, and (ii) shows
    the Receiver-Operating-Characteristics curve for the predicted links.
                    
    Example:
    ------------------
    reconstruction(14,500) reconstructs the incoming links of neuron 14 using
    500 recorded events.
    
    """
    
    pl.close("all")
    pl.style.use("seaborn-pastel")

    #Reading network parameters
    J=np.loadtxt("Data/connectivity.dat")
    Ad=np.copy(J)
    Ad[Ad!=0]=1
    delay=np.loadtxt("Data/delay.dat")
    delay=delay[0]
    
    #Reading spike trains from file
    N=len(Ad)
    spk= [[] for x in xrange(N)]
    ISIs= [[] for x in xrange(N)]
    ex_spk=np.loadtxt("Data/ex_neurons-%d-0.gdf"%(N+1))
    in_spk=np.loadtxt("Data/in_neurons-%d-0.gdf"%(N+2))
    
    start_time = time.time()
    
    #Sorting spike trains into a list of lists (numbering is according simulation files)
    for i in range(len(ex_spk)):
    	spk[int(ex_spk[i,0])-1].append(ex_spk[i,1])
    
    for i in range(len(in_spk)):
    	spk[int(in_spk[i,0])-1].append(in_spk[i,1])
    
    #Computing Interspike Intervals
    for i in range(N):
    	ISIs[i]=(np.array(spk[i][1:])-np.array(spk[i][:-1])).tolist()
    
    #Computing Cross-spike Intervals for selected neuron
    i=neuron
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
    
    #Determining maximum number of Cross-spike Intervals per event                
    K_events= []
    for ti in range(len(spk[i])-1):
    	K_list=[]	
    	for j in range(N):
    		K_list.append(len(events[ti][j]))
    	K_events.append(max(K_list))
    
    #Constructing the events
    ISEs=np.zeros((max(K_events)*N,len(K_events)))
    for ti in range(len(K_events)):
    	for j in range(N):
    		a=events[ti][j]
    		k=len(a)
    		for l in range(k):
    			ISEs[l*N+j,ti]=a[l]
    ISIsi=np.asarray(ISIs[i])
    d=np.vstack((ISEs,ISIsi))
    
    #Determing the reference event
    D=sklearn.metrics.pairwise.euclidean_distances(d.T)
    center=[]
    for ti in range(len(K_events)):
    	center.append(np.mean(D[ti,:]))
    center_index=np.argmin(center)
    
    #Determing order in an increasing manner with respect to the reference event
    non_ranked=D[center_index,:]
    closest_index=np.argsort(non_ranked,axis=0)
    
    #Constructing the system of equations
    c=ISEs[:,center_index].reshape(max(K_events)*N,1)
    W=ISEs-c*np.ones((1,len(K_events)))
    y=ISIsi.reshape((1,len(K_events)))-ISIsi[center_index]*np.ones((1,len(K_events)))
    X=np.copy(W)
    K_vector=np.asarray(K_events)
    K_vector=K_vector[closest_index]
    
    #Ordering the system of equations according to distance with respect to the
    #referenc event
    k=K_vector[M]
    y=y[0,closest_index]
    y=y.reshape((1,len(K_events)))
    X=X[:,closest_index]
    
    print " "
    print "Problem's Characteristics "
    print "==================================="
    print "Unit: %d"%i
    print "Maximum number of spikes during ISI: %d"%max(K_events)
    print "Coefficient of variation: %f"%(np.std(ISIsi)/np.mean(ISIsi))
    print "Skewness of ISIs: %f"%scipy.stats.skew(ISIsi)
    print "Kurtosis of ISIs: %f"%scipy.stats.kurtosis(ISIsi)
    print "Maximum number of spikes at center: %d"%K_events[center_index]
    print "Interspike interval at center: %f"%ISIsi[center_index]
    print "Network size: %d"%N
    print "Number of incoming connections: %d"%np.linalg.norm(Ad[i,:],1)
    print "Number of unknowns: %d"%(k*N)
    print "==================================="
    print " "
    
    #Selecting up to M events to solve the system of equations
    y=y[:,1:M+1]
    X=X[:k*N,1:M+1]

    #Solving the system of equations    
    print "Employing L2 norm optimization"
    g=np.dot(y,np.linalg.pinv(X))
    
    #Selecting only the first firing profile as connectivity proxy
    G=np.reshape(g,(k,-1))
    H=G[0,:]
    
    #Computing reconstruction quality
    fpr, tpr, thresholds = metrics.roc_curve(np.fabs(Ad[i,:]),np.fabs(H),pos_label=1)
    
    print " "
    print "Predictions"
    print "==================================="
    print "AUC score: %f"%metrics.auc(fpr, tpr)
    print "Equations: %d"%M
    print "==================================="
    
    print("--- %s seconds ---" % (time.time() - start_time))
    
    #Plotting results
    f, axarr = pl.subplots(1,2,figsize=(8,4))
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

#Running an example
neuron=41
ISIs=500
reconstruction(neuron,ISIs)