from .cache import *
import threading, copy

class ColorCache(Cache):
    def __init__(self, colorId, sizeRatio, maxSize):
#         Cache.__init__(self, maxLFUCacheSize + maxLRUCacheSize)
        self.cacheLFU = LFUCache(sizeRatio * maxSize)
        self.sizeRatio = sizeRatio
        
        if sizeRatio < 1.0:
            self.cacheLRU = LRUCache(int((1-sizeRatio) * maxSize))   
        else:
            self.cacheLRU = LRUCache(0)
            
        self.colorId = colorId
        self.maxSize = maxSize
        
    def setServerColor(self, colorId):
        self.colorId = colorId
        
    def set(self, fileId, fileColor, fileSize):
        fileSize = int(fileSize)
        if fileSize > self.maxSize:
            return
        if int(self.colorId,2) & int(fileColor,2) != 0:
            self.cacheLFU.set(fileId, fileSize)
        else:
            self.cacheLRU.set(fileId, fileSize)
            
            
    def get(self, fileId, fileColorId):
        if int(self.colorId, 2) & int(fileColorId ,2) != 0:
            return self.cacheLFU.get(str(fileId))
        else:
            return self.cacheLRU.get(fileId)

    def getWithoutCount(self, fileId, fileColorIds):
        return self.get(fileId)
            
    def clone(self):
        cloneCache = ColorCache(self.colorId, self.sizeRatio, self.maxSize)
        
        
        cloneCache.cacheLFU.bucket_head = copy.deepcopy(self.cacheLFU.bucket_head)
        cloneCache.cacheLFU.cache = copy.deepcopy(self.cacheLFU.cache)
        cloneCache.cacheLFU.currentSize = self.cacheLFU.currentSize
        
        cloneCache.cacheLRU.hashMap = copy.deepcopy(self.cacheLRU.hashMap)
        cloneCache.cacheLRU.head, cloneCache.cacheLRU.end = self.cacheLRU.cloneHeadAndEnd()
        cloneCache.cacheLRU.currentSize = self.cacheLRU.currentSize
        
        
        return cloneCache 