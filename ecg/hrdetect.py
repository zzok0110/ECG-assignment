# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 19:14:46 2022

@author: Zzera
"""

import numpy as np
import pylab as pl
import scipy.signal as signal
import firfilter
from firfilter import FIRfilter



def get_maxima(values:np.ndarray):
    max_index=signal.argrelmax(values)[0]
    return max_index#, values[max_index]

    
fs = 1000
M = 1000
data =np.loadtxt('ECG_1000Hz_4.dat')
y = data
yf= np.fft.fft(y)
cutoff1 = 45
cutoff2 = 55
cutoff = 3
 
h1 = firfilter.FIRfiltercoefficients.bandstopDesign(fs,cutoff1,cutoff2)
filter1 = firfilter.FIRfilter(h1)
h2 = firfilter.FIRfiltercoefficients.highpassDesign(fs,cutoff)
filter2 = firfilter.FIRfilter(h2)
   
f1 = np.zeros(len(y))
f2 = np.zeros(len(y))

for i in range(len(y)):
    f1[i]= filter2.dofilter(y[i]) #qu jixian gaotong
for i in range(len(y)):
    f2[i] = filter1.dofilter(f1[i]) #zai qu 50hz 
 
template = f2[2500:3000]
fir_coeff = template[::-1]
filter3 = FIRfilter(fir_coeff)
det = np.zeros(len(y))

for i in range(len(y)):
    det[i] = filter3.dofilter(f2[i])
    det[i] = det[i]*det[i]

pl.figure(1) 
pl.title('template')
pl.xlabel('time/ms')
pl.ylabel('ECG')
pl.plot(template)
pl.savefig("template.svg",dpi=300)
pl.figure(2)
pl.title('real R peak')
pl.xlabel('time/ms')
pl.ylabel('ECG')
pl.plot(f2[1000:3000])
pl.savefig("real R peak.svg",dpi=300)

pl.figure(3)
pl.title('R peaks in ECG')
pl.xlabel('time/ms')
pl.plot(det)
pl.savefig("R peaks in ECG.svg",dpi=300)

new=np.zeros(len(det))
for i in range(len(det)):
    if det[i]>20:
        new[i]=det[i]

#print(get_maxima(new))

a=get_maxima(new)
T=np.zeros(len(a)-1)

for i in range(len(a)-1):
    T[i]=a[i+1]-a[i]
#print(T)
inverse_T=1/T



ECG=np.zeros(len(det))

b=0
for j in range(len(T)-1):
    b=int(T[j])+b
    ECG[b:b+int(T[j+1])]=inverse_T[j]


pl.figure(4)
pl.title('momentary heartrate')
pl.xlabel('time/ms')
pl.ylabel(' inverse intervals between R-peaks')
pl.plot(ECG)
pl.savefig("momentary heartrate.svg",dpi=300)