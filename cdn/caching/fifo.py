from .cache import *
import copy

class FIFOCache(Cache):
    def __init__(self, maxCacheSize=1000000):
        Cache.__init__(self, maxCacheSize)
        self.queue = []
    def get(self, fileId):
        return self.exist(fileId)
        
        
    def exist(self, fileId):
        for curfileId, curFileSize in self.queue:
            if curfileId == fileId:
                self.stats.hit += 1
                return curFileSize
            
        self.stats.miss += 1
        return -1
        
    def set(self, fileId, fileSize):
        if self.exist(fileId) != -1:
            return
        
        while self.currentSize + fileSize > self.maxSize:
            curfileId, curfileSize = self.queue.pop(0)
            self.currentSize -= curfileSize
            
        self.queue.append((fileId, fileSize))
        self.currentSize += fileSize