#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 15:44:05 2024

@author: shahab-nasiri
"""

import time
import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates

t = datetime(2022,4,22)
T1 = datetime(2023,1,1)
T2 = datetime(2023,3,31)
Q123_days = ((T2 - T1).days+1)

# Initialise parameters
S0 = 135.89               # initial stock price
K = 165.00                # strike price
T = ((T2-t).days)/365     # time to maturity in years
r = 0.0                   # annual risk-free rate
vol = 0.5307              # volatility (%)

N = Q123_days*288         # number of time steps
M = 1000                  # number of simulations

# slow steps - discretized every day
N = Q123_days

T_tT2 = ((T2-t).days+1)/365
T_tT1 = (T1-t).days/365
print("Start averaging from T1", round(T_tT1,2), "to T2", round(T_tT2,2))

obs_times = np.linspace(T_tT1,T_tT2,N+1)
# Include starting time, uneven time delta's
obs_times[0]=0
dt = np.diff(obs_times)
print("Number of time steps:", len(dt))

start_time = time.time()

nudt = np.full(shape=(N), fill_value=0.0)
volsdt = np.full(shape=(N), fill_value=0.0)

# Precompute constants
for i in range(N):
    nudt[i] = (r - 0.5*vol**2)*dt[i]
    volsdt[i] = vol*np.sqrt(dt[i])

# Standard Error Placeholders
sum_CT = 0
sum_CT2 = 0

# Monte Carlo Method
for i in range(M):

    St = S0
    At_sum = 0

    for j in range(N):
        epsilon = np.random.normal()
        St = St*np.exp( nudt[j] + volsdt[j]*epsilon )
        At_sum += St

    A = At_sum/N
    CT = max(0, A - K)

    sum_CT = sum_CT + CT
    sum_CT2 = sum_CT2 + CT*CT

# Compute Expectation and SE
C0_slow = np.exp(-r*T)*sum_CT/M
sigma = np.sqrt( (sum_CT2 - sum_CT*sum_CT/M)*np.exp(-2*r*T) / (M-1) )
SE_slow = sigma/np.sqrt(M)

time_comp_slow = round(time.time() - start_time,4)
print("Call value is ${0} with SE +/- {1}".format(np.round(C0_slow,3),np.round(SE_slow,3)))
print("Computation time is: ", time_comp_slow)

T_tot = (T2-t).days/365
T_start = (T1-t).days/365
M = 50
N = (T2-t).days

# precompute constants
dt = T/N
nudt = (r - 0.5*vol**2)*dt
volsdt = vol*np.sqrt(dt)

# Monte Carlo Method
Z = np.random.normal(size=(N, M))
delta_St = nudt + volsdt*Z
ST = S0*np.cumprod( np.exp(delta_St), axis=0)
ST = np.concatenate( (np.full(shape=(1, M), fill_value=S0), ST ) )

fig, ax = plt.subplots(figsize=(8,6))

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "16"

tx = pd.date_range(start=t,end=T2)
ax.plot(tx,ST)
ax.plot([T1, T1],[np.amin(ST),np.amax(ST)], linewidth=5, label='$T_1$')
ax.plot([T2, T2],[np.amin(ST),np.amax(ST)], linewidth=5, label='$T_2$')

# convert to matplotlib date representation
start = mdates.date2num(T1)+1
end = mdates.date2num(T2)-1
width = end - start

ax.add_patch(Rectangle((start,np.amin(ST)),width,np.amax(ST)-np.amin(ST),
                       alpha=0.5,
                    facecolor='grey',
                    lw=4, label='Average'))
plt.xlabel('Time')
plt.ylabel('Forward Price')
plt.title('Average Rate Base Load Quarterly Options')

plt.legend()
plt.show()