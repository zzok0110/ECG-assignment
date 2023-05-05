# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 18:51:23 2022

@author: Zzera
"""

import numpy as np
import pylab as pl
import firfilter

fs = 1000
M = 1000
data =np.loadtxt('ECG_1000Hz_4.dat')
y = data
yf= np.fft.fft(y)
f3 = np.zeros(len(y))
cutoff1 = 45
cutoff2 = 55

h1 = firfilter.FIRfiltercoefficients.bandstopDesign(fs,cutoff1,cutoff2)
filter1 = firfilter.FIRfilter(h1)

for i in range(len(y)):
   f3[i] = filter1.dofilter(y[i]) # task3 duibi

NTAPS = 100
LEARNING_RATE = 0.001
noise = 50
f = firfilter.FIRfilter(np.zeros(NTAPS))

c=f3.shape[0]
cfft=np.abs(np.fft.rfft(f3)/c)
cfreqs=np.linspace(0,fs/2,int(c/2)+1)

pl.figure(1)
pl.xlim(0,2000)
pl.title('bandstop filter to remove the 50 Hz')
pl.xlabel('Time/ms')
pl.ylabel('ECG')
pl.plot(f3)
pl.savefig("bandstop filter to remove the 50 Hz.svg")   

t3 = np.empty(len(y))
for i in range(len(y)):
    ref_noise = np.sin(2.0 * np.pi * noise/fs * i)
    t3[i] = f.doFilterAdaptive(y[i], ref_noise, LEARNING_RATE)
    
    
d=t3.shape[0]
dfft=np.abs(np.fft.rfft(t3)/d)
dfreqs=np.linspace(0,fs/2,int(d/2)+1)

pl.figure(2)
pl.title('bandstop filter spectrum')
pl.xlabel('Freq/Hz')
pl.ylabel('Amplitude')
pl.xlim(0,200)
pl.plot(cfreqs,cfft)
pl.savefig("bandstop filter spectrum.svg")

pl.figure(3)
pl.xlim(0,2000)
pl.title('adaptive LMS filter to remove the 50 Hz')
pl.xlabel('Time/ms')
pl.ylabel('ECG')
pl.plot(t3)
pl.savefig("adaptive LMS filter to remove the 50 Hz.svg")   

pl.figure(4)
pl.title('adaptive LMS filter spectrum')
pl.xlabel('Freq/Hz')
pl.ylabel('Amplitude')
pl.xlim(0,200)
pl.plot(dfreqs,dfft)
pl.savefig("adaptive LMS filter spectrum.svg")

pl.show()


