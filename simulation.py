#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 21:40:40 2017

simulation.py generates time series of randomly-connected ensembles of excitatory
and inhibitory leaky integrate-and-fire neurons with delta synapses.

This code is based on the example "random balanced network (delta synapses)" of
the Nest-Simulator documentation:

    http://www.nest-simulator.org/py_sample/brunel_delta_nest/index.html

Output
----------
"Data/delay.dat" :  Contains the synaptic delay employed in the simulations

"Data/ex_neurons" : Contains the corresponding spiking times of excitatory neurons

"Data/in_neurons" : Contains the corresponding spiking times of inhibitory neurons

"Data/connectivity.dat" :  Contains the connecitivy matrix used in the simulation

Accompanying material to "Inferring network connectivity from event timing
patterns".

@author: Jose Casadiego

Tested on: NEST 2.14.0
"""

#import readline
import nest
import nest.raster_plot
import time
import numpy as np
import os as os

###############################################################################
#Randomization of dynamics
###############################################################################
nest.ResetKernel()
msd = int(np.ceil(100000*np.random.rand(1)))
N_vp = nest.GetKernelStatus(['total_num_virtual_procs'])[0]
pyrngs = [np.random.RandomState(s) for s in range(msd, msd+N_vp)]
nest.SetKernelStatus({'grng_seed' : msd+N_vp})

###############################################################################
# Determining the path for output files
###############################################################################
os.system("rm -r Data")
os.system("mkdir Data")
 
nest.ResetKernel()

startbuild = time.time()

###############################################################################
# Definining simulation parameters
###############################################################################
dt = 0.1  # time resolution in ms
simtime = 50000.0  # total simulation time in ms
delay = 1.5  # synaptic delay in ms
g = 4.0  # ratio inhibition/excitation
epsilon = 0.1  # connection probability
J = 0.3  # postsynaptic amplitude in mV
J_ex = J  # amplitude of excitatory postsynaptic potential
J_in = -g * J_ex  # amplitude of inhibitory postsynaptic potential
np.savetxt("Data/delay.dat",np.ones((3,1))*delay,delimiter="\t",fmt="%1.4f")

###############################################################################
# Defining number of neurons and synapses
###############################################################################
order = 20
NE = 4 * order  # number of excitatory neurons
NI = 1 * order  # number of inhibitory neurons
N_neurons = NE + NI  # number of neurons in total
CE = int(epsilon * NE)  # number of excitatory synapses per neuron
CI = int(epsilon * NI)  # number of inhibitory synapses per neuron
C_tot = int(CI + CE)  # total number of synapses per neuron

###############################################################################
# Defining default properties of neurons
###############################################################################
tauMem = 20.0  # time constant of membrane potential in ms
theta = 20.0  # membrane threshold potential in mV
neuron_params = {"C_m": 1.0,
                 "tau_m": tauMem,
                 "t_ref": 2.0,
                 "E_L": 0.0,
                 "V_reset": 0.0,
                 "V_m": 0.0,
                 "V_th": theta}

###############################################################################
# Creating neurons and spike detectors
###############################################################################
print("Building network model")
nest.SetKernelStatus({"resolution": dt, "print_time": True,
                      "overwrite_files": True})

nest.SetDefaults("iaf_psc_delta", neuron_params)

nodes_ex = nest.Create("iaf_psc_delta", NE)
nodes_in = nest.Create("iaf_psc_delta", NI)
espikes = nest.Create("spike_detector")
ispikes = nest.Create("spike_detector")

###############################################################################
# Randomizing initial conditions and external inputs of neurons
###############################################################################
for neuron in nodes_ex:
    nest.SetStatus([neuron], {"V_m": 0.0+(theta-0.0)*np.random.rand()})
    nest.SetStatus([neuron], {"I_e": 1.0*(1.2+(1.4-1.2)*np.random.rand())})

for neuron in nodes_in:
    nest.SetStatus([neuron], {"V_m": 0.0+(theta-0.0)*np.random.rand()})
    nest.SetStatus([neuron], {"I_e": 1.0*(1.2+(1.4-1.2)*np.random.rand())})

###############################################################################
# Defining output files
###############################################################################
nest.SetStatus(espikes, [{"label": "Data/ex_neurons",
                          "withtime": True,
                          "withgid": True,
                          "to_file": True}])
 
nest.SetStatus(ispikes, [{"label": "Data/in_neurons",
                          "withtime": True,
                          "withgid": True,
                          "to_file": True}])

###############################################################################
# Connecting neurons and spike detectors
###############################################################################
print("Connecting devices")

nest.CopyModel("static_synapse", "excitatory",
               {"weight": J_ex, "delay": delay})

nest.CopyModel("static_synapse", "inhibitory",
               {"weight": J_in, "delay": delay})

nest.Connect(nodes_ex, espikes, syn_spec="excitatory")
nest.Connect(nodes_in, ispikes, syn_spec="excitatory")
 
sources_ex = np.random.random_integers(1, NE, (N_neurons, CE))
sources_in = np.random.random_integers(NE + 1, N_neurons, (N_neurons, CI))

for n in range(N_neurons):
    nest.Connect(list(sources_ex[n]), [n + 1], syn_spec="excitatory")

for n in range(N_neurons):
    nest.Connect(list(sources_in[n]), [n + 1], syn_spec="inhibitory")
    
###############################################################################
# Extracting the connectivity matrix
###############################################################################
connectivity=np.zeros((N_neurons,N_neurons))

conn_ex=nest.GetConnections(nodes_ex)
conn_ex_source= nest.GetStatus(conn_ex, keys='source')
conn_ex_target= nest.GetStatus(conn_ex, keys='target')
conn_ex_weight= nest.GetStatus(conn_ex, keys='weight')

conn_in=nest.GetConnections(nodes_in)
conn_in_source= nest.GetStatus(conn_in, keys='source')
conn_in_target= nest.GetStatus(conn_in, keys='target')
conn_in_weight= nest.GetStatus(conn_in, keys='weight')

for i in range(len(conn_ex_source)):
	if conn_ex_source[i]<= N_neurons and conn_ex_target[i]<= N_neurons:
		connectivity[conn_ex_source[i]-1,conn_ex_target[i]-1]=conn_ex_weight[i]
for i in range(len(conn_in_source)):
	if conn_in_source[i]<=N_neurons and conn_in_target[i]<= N_neurons:
		connectivity[conn_in_source[i]-1,conn_in_target[i]-1]=conn_in_weight[i]
		
connectivity=connectivity.T
np.savetxt("Data/connectivity.dat",connectivity,delimiter="\t",fmt="%1.4f")

endbuild = time.time()

###############################################################################
# Running the simulation
###############################################################################

print("Simulating")
 
nest.Simulate(simtime)

endsimulate = time.time()

events_ex = nest.GetStatus(espikes, "n_events")[0]
events_in = nest.GetStatus(ispikes, "n_events")[0]

rate_ex = events_ex / simtime * 1000.0 / NE
rate_in = events_in / simtime * 1000.0 / NI

num_synapses = (nest.GetDefaults("excitatory")["num_connections"] +
                nest.GetDefaults("inhibitory")["num_connections"])

build_time = endbuild - startbuild
sim_time = endsimulate - endbuild

print("Brunel network simulation (Python)")
print("Number of neurons : {0}".format(N_neurons))
print("       Exitatory  : {0}".format(int(CE * N_neurons)))
print("       Inhibitory : {0}".format(int(CI * N_neurons)))
print("Excitatory rate   : %.2f Hz" % rate_ex)
print("Inhibitory rate   : %.2f Hz" % rate_in)
print("Building time     : %.2f s" % build_time)
print("Simulation time   : %.2f s" % sim_time)

