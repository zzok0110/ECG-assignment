# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 18:49:48 2022

@author: Zzera
"""

import numpy as np


fs = 1000
M = 1000
import pylab as pl
data =np.loadtxt('ECG_1000Hz_4.dat')
y = data
yf = np.fft.fft(y)


class FIRfiltercoefficients:
    def bandstopDesign(fs,cutoff1,cutoff2):
        M=fs
        k1 = int(cutoff1/fs * M)
        k2 = int(cutoff2/fs * M)
        X = np.ones(M)
        X[k1:k2+1] = 0
        X[M-k2: M-k1+1] = 0
        x = np.fft.ifft(X)
        x = np.real(x)
        h = np.zeros(M)
        h[0:int(M/2)] = x[int(M/2):M]
        h[int(M/2):M] = x[0:int(M/2)]
        h1 = h * np.hamming(M)
        return(h1)
  
    def highpassDesign(fs,cutoff):
        M=fs
        k = int(cutoff/fs * M)
        X = np.ones(M)
        X[0:k+1] = 0
        X[M-k:M+1] = 0
        x = np.fft.ifft(X)
        x = np.real(x)
        h = np.zeros(M)
        h[0:int(M/2)] = x[int(M/2):M]
        h[int(M/2):M] = x[0:int(M/2)]
        h2 = h * np.hamming(M)
        return(h2)

#task2
class FIRfilter:
    def __init__(self,_coefficients):
        self.ntaps =len(_coefficients)
        self.coefficients = _coefficients
        self.buffer =np.zeros(self.ntaps)
    def dofilter(self,v):
        for j in range(self.ntaps-1):
            self.buffer[self.ntaps-j-1] = self.buffer[self.ntaps-j-2]
        self.buffer[0] = v
        return np.inner(self.buffer,self.coefficients)
    def doFilterAdaptive(self, signal, ref_noise, learningRate):
        canceller = self.dofilter(ref_noise)
        output_signal = signal - canceller
        for j in range(self.ntaps):
            self.coefficients[j] = self.coefficients[j] + output_signal * learningRate * self.buffer[j]
        return output_signal


if __name__ == '__main__':   
  pl.figure(1)    
  pl.title('ECG')
  pl.plot(data)
  pl.xlabel('time/msec')
  pl.ylabel('ECG/RAW')
  pl.savefig('Raw ECG pattern.svg')
  pl.show()
  h1 = FIRfiltercoefficients.bandstopDesign(1000,45,55) #50Hz removal
  pl.figure(2)
  pl.title('H of Bandstop Filter')
  pl.xlabel('n')
  pl.ylabel('H(n)')
  pl.plot(h1)
  pl.savefig('h of bandstop filter.svg')
    
  h2 = FIRfiltercoefficients.highpassDesign(1000,50) #50Hz removal
  pl.figure(3)
  pl.title('H of Highpass Filter')
  pl.xlabel('n')
  pl.ylabel('H(n)')
  pl.plot(h2)
  pl.savefig('h of highpass filter.svg')
    
    