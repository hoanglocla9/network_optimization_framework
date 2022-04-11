import math
import networkx as nx
from src.util.gen_files import * 
from src.util.virtual_run import *

def caculate_tail(S, C, totalContents):
    N = len(S)
    result = 0
    
    if N * C > totalContents:
        result = totalContents
    else:
        result = N * C

    for i in range(0, N-1):
        result -= S[i]
        
    return result

def get_hop_count(graph, user_i, cache_j):
    return nx.dijkstra_path(graph,source=user_i,target=cache_j)
    
def estimate_traffic(S_tmp, graph, 
                     nearestColorServerInfo , totalColor, uniqueSortedContentList, cacheDict,
                    serverToColorMap, fileSize, routingTable, warmUpReqDict, runReqDict, clientList):
    contentToColorDict = colorizeWithSeparatorRanks(uniqueSortedContentList, S_tmp, totalColor)
    estimasted_traffic = runWithColorRouting(graph, cacheDict, contentToColorDict, nearestColorServerInfo, 
                               serverToColorMap, fileSize, routingTable, runReqDict, clientList)
    return estimasted_traffic

def evaluate_traffic(S_tmp, graph, 
                     nearestColorServerInfo , totalColor, uniqueSortedContentList, cacheDict,
                    serverToColorMap, fileSize, routingTable, warmUpReqDict, runReqDict, clientList):
    contentToColorDict = colorizeWithSeparatorRanks(uniqueSortedContentList, S_tmp, totalColor)
    warmUpColor(graph, cacheDict, contentToColorDict, nearestColorServerInfo, serverToColorMap, fileSize, routingTable, warmUpReqDict, clientList)
    estimasted_traffic = runWithColorRouting(graph, cacheDict, contentToColorDict, nearestColorServerInfo, 
                               serverToColorMap, fileSize, routingTable, runReqDict, clientList)
    return estimasted_traffic


def isSeparatorRanksValid(S):
    for i in range(1, len(S)):
        if S[i-1] > S[i]:
            return False
        
    return True


def cloneCacheDict(cacheDict):
    result = {}
    for cacheId in cacheDict:
        result[cacheId] = cacheDict[cacheId].clone()
    return result   


def compute_rank(numberOfColor, cacheServerCapacity, fileSize, graph, nearestColorServerInfo, contentGenerator,
        cacheDict, serverToColorMap, warmUpReqNums, runReqNums, clientList, increaseSRs=1, interval="Interval0", parallel_idx=0):

    N = int(numberOfColor)
    C = int(cacheServerCapacity/ fileSize)
    if fileSize < 0:
        print(fileSize)
        raise Exception('File size must be a positive number for color based algorithms')
    numberOfCache = len(cacheDict.keys())
    numberOfContent = len(contentGenerator.uniqueSortedContentList[interval])
    incr = int(increaseSRs * numberOfContent / 100.0)
        
    S = [0] * N
    S_prev = [0] * N
    T_min = float("inf")
    if N*C > numberOfContent:
        S[N-1] = int(numberOfContent)
    else:
        S[N-1] = N * C
    routingTable = {}
    
    warmUpReqDict = {}
    runReqDict = {}
    tempCacheDict = cloneCacheDict(cacheDict)
    
    if contentGenerator.dist != None:
        for client in clientList:
            warmUpReqDict[client] = contentGenerator.randomGen(warmUpReqNums)
            runReqDict[client] = contentGenerator.randomGen(runReqNums)
        # Fill cache
        initContentToColorDict = colorizeWithSeparatorRanks(contentGenerator.uniqueSortedContentList[interval], S, numberOfColor)
        warmUpColor(graph, tempCacheDict, initContentToColorDict, nearestColorServerInfo, serverToColorMap, fileSize, routingTable, warmUpReqDict, clientList)
    
    else:
        intervalIdx = int(interval.replace("Interval", ""))
        if intervalIdx == 0 or interval == "noInterval":
            for cache in contentGenerator.custom_data:
                client = cache.replace("Cache_", "client_")
                warmUpReqDict[client] = contentGenerator.custom_data[cache][interval]
                runReqDict[client] = contentGenerator.custom_data[cache][interval]
            # Fill cache
            initContentToColorDict = colorizeWithSeparatorRanks(contentGenerator.uniqueSortedContentList[interval], S, numberOfColor)
            warmUpColor(graph, tempCacheDict, initContentToColorDict, nearestColorServerInfo, serverToColorMap, fileSize, routingTable, warmUpReqDict, clientList)
        else:
            for cache in contentGenerator.custom_data:
                client = cache.replace("Cache_", "client_")
                runReqDict[client] = contentGenerator.custom_data[cache][interval]
                
    
    with open("data/saveCacheDict_" + str(parallel_idx) + ".pkl", "wb") as f:
        pickle.dump(tempCacheDict, f)
        
    result = {}
    while S_prev != S:
        S_prev = list(S)
        for i in range(0, N-1):
            for v in range(S[max(1, i)-1], S[i+1]+1, incr):
                S_tmp = list(S)
                S_tmp[i] = v
                S_tmp[N - 1] = caculate_tail(S_tmp, C, numberOfContent)
                if not isSeparatorRanksValid(S_tmp):
                    break

                with open("data/saveCacheDict_" + str(parallel_idx) + ".pkl", "rb") as f:
                    tempCloneDict = pickle.load(f)

                T_est = estimate_traffic(S_tmp, graph, nearestColorServerInfo, N, 
                                         contentGenerator.uniqueSortedContentList[interval], tempCloneDict, 
                                         serverToColorMap, fileSize, routingTable, warmUpReqDict, runReqDict, 
                                         clientList)
                if T_est < T_min:
                    T_min = T_est
                    S = S_tmp
                    result = {"cacheDict": tempCloneDict, "S": S}
    
    return result


def estimate_traffic_shortest_path_with_color(S_tmp, graph, 
                     nearestColorServerInfo , totalColor, uniqueSortedContentList, cacheDict, fileSize, routingTable, runReqDict, clientList):
    contentToColorDict = colorizeWithSeparatorRanks(uniqueSortedContentList, S_tmp, totalColor)
    estimasted_traffic = runWithShortestPath(graph, cacheDict, fileSize, "tag-color", routingTable, runReqDict, clientList, contentToColorDict)
    return estimasted_traffic

def evaluate_traffic_shortest_path_with_color(S_tmp, graph, 
                     nearestColorServerInfo , totalColor, uniqueSortedContentList, cacheDict, fileSize, routingTable, warmUpReqDict, runReqDict, clientList):
    contentToColorDict = colorizeWithSeparatorRanks(uniqueSortedContentList, S_tmp, totalColor)
    warmUpCacheShortestPath(graph, cacheDict, fileSize, "tag-color", routingTable, warmUpReqDict, clientList, contentToColorDict)
    estimasted_traffic = runWithShortestPath(graph, cacheDict, fileSize, "tag-color", routingTable, runReqDict, clientList, contentToColorDict)
    return estimasted_traffic

def compute_rank_shortest_path_with_color(numberOfColor, cacheServerCapacity, fileSize, graph, nearestColorServerInfo,                                                         contentGenerator, cacheDict, warmUpReqNums, runReqNums, 
                                          clientList, increaseSRs=1, interval="", parallel_idx=0):
    
    N = numberOfColor
    C = int(cacheServerCapacity / fileSize)
    if fileSize < 0:
        print(fileSize)
        raise Exception('File size must be a positive number for color based algorithms')
        
    numberOfCache = len(cacheDict.keys())
    numberOfContent = len(contentGenerator.uniqueSortedContentList[interval])
        
    S = [0] * N
    S_prev = [0] * N
    T_min = float("inf")
    
    if N*C > numberOfContent:
        S[N-1] = numberOfContent
    else:
        S[N-1] = N * C
    incr = int(increaseSRs * numberOfContent / 100.0)
    routingTable = {}
    
    warmUpReqDict = {}
    runReqDict = {}
    tempCacheDict = cloneCacheDict(cacheDict)
    
    if contentGenerator.dist != None:
        for client in clientList:
            warmUpReqDict[client] = contentGenerator.randomGen(warmUpReqNums)
            runReqDict[client] = contentGenerator.randomGen(runReqNums)
        initContentToColorDict = colorizeWithSeparatorRanks(contentGenerator.uniqueSortedContentList[interval], S, numberOfColor)
        warmUpCacheShortestPath(graph, tempCacheDict, fileSize, "tag-color", routingTable, warmUpReqDict, clientList, initContentToColorDict)
    else:
        intervalIdx = int(interval.replace("Interval", ""))
        if intervalIdx == 0 or interval == "noInterval":
            for cache in contentGenerator.custom_data:
                client = cache.replace("Cache_", "client_")
                warmUpReqDict[client] = contentGenerator.custom_data[cache][interval]
                runReqDict[client] = contentGenerator.custom_data[cache][interval]
            initContentToColorDict = colorizeWithSeparatorRanks(contentGenerator.uniqueSortedContentList[interval], S, numberOfColor)
            warmUpCacheShortestPath(graph, tempCacheDict, fileSize, "tag-color", routingTable, warmUpReqDict, clientList, initContentToColorDict)
        else:
            for cache in contentGenerator.custom_data:
                client = cache.replace("Cache_", "client_")
                runReqDict[client] = contentGenerator.custom_data[cache][interval]
    
    with open("data/saveCacheDict_" + str(parallel_idx) + ".pkl", "wb") as f:
        pickle.dump(tempCacheDict, f)
        
    result = {}
    while S_prev != S:
        S_prev = list(S)
        for i in range(0, N-1):
            for v in range(S[max(1, i)-1], S[i+1]+1, incr):
                S_tmp = list(S)
                S_tmp[i] = int(v)
                S_tmp[N - 1] = int(caculate_tail(S_tmp, C, numberOfContent))
                if not isSeparatorRanksValid(S_tmp):
                    break
                with open("data/saveCacheDict_" + str(parallel_idx) + ".pkl", "rb") as f:
                    tempCloneDict = pickle.load(f)

                T_est = estimate_traffic_shortest_path_with_color(S_tmp, graph, nearestColorServerInfo, N, contentGenerator.uniqueSortedContentList[interval], tempCloneDict, fileSize, routingTable, runReqDict, clientList)
                if T_est < T_min:
                    T_min = T_est
                    S = S_tmp
                    result = {"cacheDict": tempCloneDict, "S": S}
    return result 