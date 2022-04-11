import networkx as nx 
import pickle, random, os, shutil
from problem.network_optimization.util.gen_files import *
from problem.network_optimization.caching.cache import *
from problem.network_optimization.util.virtual_run import *

def generate_cache_info(graph):
    result = {}
    for node in graph.nodes:
        result[node] = {'size': 100, 'type': 'LRU'}
    return result

class NetTopology (object):
    def __init__( self, topologyFile, mainServerId, configDirPath="tmp/", cacheDictInfo=None):
        self.graph = nx.read_gml(topologyFile)
        self.editBandwidthInfo()
        self.cacheMemoryDict = {}
        self.mainServerId = 0
        self.configDirPath = configDirPath
        if not os.path.exists(configDirPath):
            os.makedirs(configDirPath)
        else:
            shutil.rmtree(configDirPath)
            os.makedirs(configDirPath)
        if cacheDictInfo is None:
            cacheDictInfo = generate_cache_info(self.graph)
        self.cacheMemoryDict = self.buildCacheMemoryDict(cacheDictInfo)
        self.cacheDictInfo = cacheDictInfo
        self.mainServerId= mainServerId
        ### fixed with gamma distribution
        # lenght = number of content
        # fixedFileSize = size of a content
        dist = GammaDistribution(K=0.475, theta=170.6067, length=200)
        self.contentGenerator = ContentGenerator(dist=dist)
        self.warmUpReqNums = 1000
        # if cacheDictInfo is not None:
        self.warmUp()
        
    def editBandwidthInfo(self):
        for data in self.graph.edges(data=True):
            if 'id' in data[2]:
                nx.set_edge_attributes(self.graph, 100*1000, "weight")
            elif 'LinkLabel' in data[2]:
                linkLabel = data[2]['LinkLabel']
                split = linkLabel.split(" ")
                if len(split) == 3:
                    bandwidth = float(split[1]) * 1000 if split[2] == 'Gpbs' else float(split[1])
                elif len(split) == 2 and 'pbs' not in split[1]:
                    unitStr = split[1][-4:]
                    bandwidth = float(split[1][:-4]) * 1000 if unitStr == 'Gpbs' else float(split[1][:-4])
                else:
                    raise( f"[ERROR] Error in LINKLABEL format!!!: {linkLabel}")
                    break
                nx.set_edge_attributes(self.graph, bandwidth, "weight")
            else:
                raise( f"[ERROR] Error in LINKLABEL format!!!: {linkLabel}")
                break
        
    def buildCacheMemoryDict(self, infoDict):
        result = {}
        if infoDict is None:
            return None
        for routerId in infoDict:
            info = infoDict[routerId]
            if info["type"] == "LRU":
                result[routerId] = LRUCache(info['size'], maxSize=info['size'])
            elif info["type"] == "LFU":
                result[routerId] = LFUCache(info['size'], maxSize=info['size'])
            elif info["type"] == "FIFO":
                result[routerId] = FIFOCache(info['size'], maxSize=info['size'])
        return result
    
    def reconfig(self, x, idx):
        newInfoDict = {}
        for idx, size in enumerate(x):
            nodeId = list(self.cacheDictInfo.keys())[idx]
            newInfoDict[nodeId] = self.cacheDictInfo[nodeId]
            newInfoDict[nodeId]['size'] = int(size)
        cacheMemoryDict = self.buildCacheMemoryDict(newInfoDict)
        self.saveCache(cacheMemoryDict, idx=idx)
        
    def warmUp(self, savedFile="tmp/warmUpReqFile.pkl"):
        if os.path.isfile(savedFile):
            with open(savedFile, "rb") as f:
                warmUpReqDict = pickle.load(f)
        else:
            warmUpReqDict = {}
            for client in self.graph.nodes:
                warmUpReqDict[client] = self.contentGenerator.randomGen(self.warmUpReqNums)

            with open(savedFile, "wb") as f:
                pickle.dump(warmUpReqDict, f)
            warmUpCacheShortestPath(self.graph, self.cacheMemoryDict, warmUpReqDict, self.mainServerId)
            self.saveCache(self.cacheMemoryDict)
            
    def saveCache(self, cacheMemoryDict, savedFile="tmp/warmUpFile.pkl", idx = None):
        if idx is not None:
            savedFile = f"tmp/warmUpFile_{idx}.pkl"
        with open(savedFile, "wb") as f:
            pickle.dump(cacheMemoryDict, f)