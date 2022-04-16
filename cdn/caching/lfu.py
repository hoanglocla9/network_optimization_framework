from .cache import *
import collections, copy

class LFUCache(Cache):
    def __init__(self, maxCacheSize=1000000, delta=10000):
        Cache.__init__(self, maxCacheSize)
        self.key_freq_map = {}

    def get(self, fileId):
        if fileId not in self.key_freq_map:
            self.stats.miss += 1
            return -1
        
        self.key_freq_map[fileId]["freq"] += 1
        self.stats.hit += 1
        return self.key_freq_map[fileId]["size"]
        
    def set(self, fileId, fileSize):
        fileSize = int(fileSize)
        if fileSize > self.maxSize:
            return
        
        if self.get(fileId) != -1:
            return 
        else:
            while self.currentSize + fileSize > self.maxSize:
                removedKey = list(self.key_freq_map.keys())[0]
                for key in self.key_freq_map:
                    if self.key_freq_map[key]["freq"] < self.key_freq_map[removedKey]["freq"]:
                        removedKey = key
                self.currentSize -= self.key_freq_map[removedKey]["size"]
                del self.key_freq_map[removedKey]
                
            self.key_freq_map[fileId] = {"freq": 1, "size": fileSize}
            self.currentSize += fileSize