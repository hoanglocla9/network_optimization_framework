import networkx as nx
import random, pickle
from .gen_files import *
routingTable = {}

def findShortestCacheServer(graph, sourceId, targetId):
    if sourceId not in routingTable:
        cacheIdPath = nx.dijkstra_path(graph, sourceId, targetId, "weight") # 
        routingTable[sourceId] = {targetId: cacheIdPath[1]}
        return cacheIdPath[1]
    else:
        if targetId not in routingTable[sourceId]:
            cacheIdPath = nx.dijkstra_path(graph, sourceId, targetId, "weight") # , "weight"
            routingTable[sourceId] = {targetId: cacheIdPath[1]}
            return cacheIdPath[1]
        else:
            return routingTable[sourceId][targetId]
    
def refreshCache(cacheDict):
    for cacheId in cacheDict:
        cacheDict[cacheId].refresh()
   

def runVirtualSendFileShortestPath(graph, cacheDict, client, contentId, mainServerId):
    traffic = 0
    nextCacheId = client
 
    if cacheDict[nextCacheId] is None:
        pass
    else:
        if cacheDict[nextCacheId].get(contentId) != -1:
            return traffic
        else:
            cacheDict[nextCacheId].set(contentId, 1)

    while True:
        routerId = findShortestCacheServer(graph, nextCacheId, mainServerId)
        traffic += 1
        if routerId == mainServerId:
            return traffic
        else:
            if cacheDict[routerId] is None:
                pass
            else:
                if cacheDict[routerId].get(contentId) == -1:
                    cacheDict[routerId].set(contentId, 1)
                else:
                    return traffic
                    
        nextCacheId = routerId
    return traffic


def warmUpCacheShortestPath(graph, cacheDict,  warmUpReqDict, mainServerId):
    idx = 0
    while True:
        isEnd = True
        for client in warmUpReqDict:
            if idx >= len(warmUpReqDict[client]):
                continue
            if client == mainServerId:
                continue
            runVirtualSendFileShortestPath(graph, cacheDict, client, warmUpReqDict[client][idx], mainServerId) ###
            isEnd = False
        if isEnd:
            break
        idx += 1
        

def runWithShortestPath(graph, cacheDict,  runReqDict, mainServerId):
    totalTraffic = 0
    idx = 0
    while True:
        isEnd = True
        for client in runReqDict:
            if idx >= len(runReqDict[client]):
                continue
            if client == mainServerId:
                continue
            oneFileTraffic = runVirtualSendFileShortestPath(graph, cacheDict, client, runReqDict[client][idx], mainServerId)
            totalTraffic += oneFileTraffic
            isEnd = False
        if isEnd:
            break
        idx += 1
    return totalTraffic