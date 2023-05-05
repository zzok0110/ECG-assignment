# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 17:35:34 2022

@author: 11582
"""
import numpy as np
import pylab as pl
import firfilter

if __name__ == '__main__':
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
     f1[i]= filter2.dofilter(y[i]) #highpass
  for i in range(len(y)):
     f2[i] = filter1.dofilter(f1[i]) #bandstop
    
 
        

pl.figure(1)
pl.plot(y)
pl.title('ECG')
pl.xlabel('Time/ms')
pl.ylabel('Amplitude')
pl.savefig('the original signal.svg')

pl.figure(2)
pl.plot(f1)
pl.title('0Hz Removed')
pl.xlabel('Time/ms')
pl.ylabel('Amplitude')
pl.savefig('0Hz removed.svg')

pl.figure(3)
pl.plot(f2)
pl.title('0Hz & 50Hz Removed')
pl.xlabel('Time/ms')
pl.ylabel('Amplitude')
pl.savefig('0&50Hz removed.svg')

y1=y.shape[0]
y1fft=np.abs(np.fft.rfft(y)/y1)
y1freqs=np.linspace(0,fs/2,int(y1/2)+1)

f11 = f1.shape[0]
f11fft=np.abs(np.fft.rfft(f1)/f11)
f11freqs=np.linspace(0,fs/2,int(f11/2)+1)

f21 = f2.shape[0]
f21fft=np.abs(np.fft.rfft(f2)/f21)
f21freqs=np.linspace(0,fs/2,int(f21/2)+1)

pl.figure(4)
pl.plot(y1freqs,y1fft)
pl.title('ECG in frequency domain')
pl.xlabel('Frequency/Hz')
pl.ylabel('Amplitude')
pl.savefig('the original signal_f.svg')

pl.figure(5)
pl.plot(f11freqs,f11fft)
pl.title('0Hz removed in frequency domain')
pl.xlabel('Frequency/Hz')
pl.ylabel('Amplitude')
pl.savefig('0Hz removed_f.svg')

pl.figure(6)
pl.plot(f21freqs,f21fft)
pl.title('0Hz & 50Hz removed in frequency domain')
pl.xlabel('Frequency/Hz')
pl.ylabel('Amplitude')
pl.savefig('0&50Hz removed_f.svg')












