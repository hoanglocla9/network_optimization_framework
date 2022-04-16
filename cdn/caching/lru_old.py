from .cache import *
import time, copy
import abc

class LRUCache(Cache):
    def __init__(self, maxCacheSize=1000000, delta=10000):
        Cache.__init__(self, maxCacheSize)
        self.hashMap = {}
        self.head = None
        self.end = None
        
    def get(self, fileId):
        if fileId not in self.hashMap:
            self.stats.miss += 1
            return -1
        
        fileNode = self.hashMap[fileId]
        self.stats.hit += 1
        if self.head == fileNode:
            return fileNode.size
        self.remove(fileNode)
        self.setHead(fileNode)
        return fileNode.size
    
    def getWithoutCount(self, fileId):
        if fileId in self.hashMap:
            return self.hashMap[fileId].size
        else:
            return -1
        
    def set(self, fileId, fileSize):
        fileSize = int(fileSize)
        if fileSize > self.maxSize:
            return
        
        if fileId in self.hashMap:
            fileNode = self.hashMap[fileId]
            fileNode.size = fileSize

            if self.head != fileNode:
                self.remove(fileNode)
                self.setHead(fileNode)
        else:
            newNode = FileNode(fileId, fileSize, freq=None, time=time.time())
            while self.currentSize + fileSize > self.maxSize:
                del self.hashMap[self.end.id]
                self.remove(self.end)
            self.setHead(newNode)
            self.hashMap[fileId] = newNode
        
    def setHead(self, fileNode):
        if not self.head:
            self.head = fileNode
            self.end = fileNode
        else:
            fileNode.prev = self.head
            self.head.next = fileNode
            self.head = fileNode
        self.currentSize += fileNode.size

    def remove(self, fileNode):
        if not self.head:
            return

        if fileNode.prev:
            fileNode.prev.next = fileNode.next
        if fileNode.next:
            fileNode.next.prev = fileNode.prev

        if not fileNode.next and not fileNode.prev:
            self.head = None
            self.end = None

        # if the node we are removing is the one at the end, update the new end
        # also not completely necessary but set the new end's previous to be NULL
        if self.end == fileNode:
            self.end = fileNode.next
            self.end.prev = None
        self.currentSize -= fileNode.size
        
    def cloneHeadAndEnd(self):
        if self.head == None and self.end == None:
            return None, None
        cloneHead = FileNode(self.head.id, self.head.size, freq=None)
        cloneEnd = None
        cloneCurrentNode = cloneHead
        currentNode = self.head.next
        
        while currentNode != None:
            newNode = FileNode(currentNode.id, currentNode.size, freq=None)
            cloneCurrentNode.next = newNode
            newNode.previous = cloneCurrentNode
            cloneCurrentNode = newNode
            cloneEnd = cloneCurrentNode
            currentNode = currentNode.next
        return cloneHead, cloneEnd