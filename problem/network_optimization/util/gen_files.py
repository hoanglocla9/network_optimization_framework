from scipy.stats import gamma
import random
import math
import operator, os
import itertools, random
import matplotlib.pyplot as plt 
import numpy as np

class ZipfDistribution:
    def __init__(self, skewness, length, sampleGen=None):
        self.denominator = 0
        self.pdfMemo = []
        self.cdfMemo = []
        self.skewness = skewness
        self.sampleGen = sampleGen
        
        self.length = length
        for i in range(1, self.length+1):
            self.denominator += 1.0 / (i ** self.skewness)

        self.pdfMemo = [1.0/ (1.0 ** self.skewness)/self.denominator]
        self.cdfMemo = [1.0/ (1.0 ** self.skewness)/self.denominator]
        
        for i in range(2, self.length+1):
            self.pdfMemo += [1.0/(float(i) ** self.skewness)/self.denominator]
            self.cdfMemo += [self.cdfMemo[i-2] + (1.0/(float(i) ** self.skewness))/self.denominator]
            
    def PDF(self, rank):
        return self.pdfMemo[rank -1]
    
    def CDF(self, rank):
        return self.cdfMemo[rank -1]
    
    def Intn(self):
        mark = random.random() 
        for i in range(1, self.length+1):
            if self.cdfMemo[i-1] > mark:
                return i
            
    def pdf(self, x): # K = a, theta = lambda
        return math.pow(x, self.K-1) * math.exp(-x/self.theta) / (math.gamma(self.K) * math.pow(self.theta, self.K))	
    
    def inverseCDF(self, C):
        if C > 1 or C < 0:
            return

        tolerance = 0.01
        x = self.length / 2
        if self.skewness != 1:
            pD = C * (12 * (self.length ** (-self.skewness + 1) - 1) / (1 - self.skewness) + 6 + 6 * (self.length ** -self.skewness) + self.skewness - self.skewness * (self.length ** (-self.skewness - 1)))
        else:
            pD = C * (12 * math.log(self.length) + 6 + 6 * (self.length ** -self.skewness) + self.skewness - self.skewness * (self.length ** (-self.skewness - 1)))
            
        while True:
            m = x ** (-self.skewness - 2)   
            mx = m * x                
            mxx = mx * x              
            mxxx = mxx * x           
            if self.skewness != 1:
                a = 12 * (mxxx - 1) / (1 - self.skewness) + 6 + 6 * mxx + self.skewness - (self.skewness * mx) - pD
            else:
                a = 12 * math.log(x) + 6 + 6 * mxx + self.skewness - (self.skewness * mx) - pD
                
            b = 12 * mxx - (6 * self.skewness * mx) + (m * self.skewness * (self.skewness + 1))
            newx = max(1, x - a / b)
            if abs(newx - x) <= tolerance:
                return round(newx)
            x = newx
        
                  
class GammaDistribution:
    def __init__(self, K, theta, length, sampleGen=None):
        self.K = K
        self.theta = theta
        self.sum = 0
        self.sampleGen = sampleGen
        self.pdfMemo = []
        self.cdfMemo = []
        self.length = length
        for i in range(1, self.length+1):
            self.sum += self.pdf(i)

        self.pdfMemo = [self.pdf(1.0)/self.sum]
        self.cdfMemo = [self.pdf(1.0)/self.sum]
        
        for i in range(2, self.length+1):
            self.pdfMemo.append(self.pdf(i)/self.sum)
            self.cdfMemo.append(self.cdfMemo[i-2]+self.pdf(i)/self.sum)
            
            
    def PDF(self, rank):
        return self.pdfMemo[rank -1]
    
    def CDF(self, rank):
        return self.cdfMemo[rank -1]
    
    def Intn(self):
        mark = random.random() 
        for i in range(1, self.length+1):
            if self.cdfMemo[i-1] > mark:
                return i
            
    def pdf(self, x): # K = a, theta = lambda
        return math.pow(x, self.K-1) * math.exp(-x/self.theta) / (math.gamma(self.K) * math.pow(self.theta, self.K))


class ContentGenerator:
    def __init__ (self, dist=None, data_path="", fixedContentSize=-1, sampleGen=None, alpha=1.0):
        self.dist = dist
        self.data_path = data_path
        self.fixedContentSize = fixedContentSize
        self.sampleGen = sampleGen
        self.custom_data = None
        self.alpha = alpha
                
    def randomGen(self, reqNums):
        if self.dist == None:
            return None
        dic = {}
        i = 0

        while i < reqNums:
            temp = str(self.dist.Intn())
            if temp in dic:
                dic[temp] += 1
            else:
                dic[temp] = 1
            i += 1
        temp = []
        for key in dic:
            for freq in range(dic[key]):
                temp.append(key)

        random.shuffle(temp)
        if self.dist.sampleGen != None:
            temp = self.dist.sampleGen.sample(temp)
        result = []
        for i in temp:
            result.append((i, self.fixedContentSize))
        return result
    

