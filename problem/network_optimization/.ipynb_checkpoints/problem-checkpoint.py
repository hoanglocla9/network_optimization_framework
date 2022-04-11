import numpy as np
from botorch.test_functions.base import (
    ConstrainedBaseTestProblem,
    MultiObjectiveTestProblem,
)
import torch
import pickle
import multiprocessing as mp
from .util.virtual_run import *
NUM_PROCESSORS = 8


def runSimulationWithPredefinedDistribution(topo, runReqNums, parallel_idx=0):
    graph = topo.graph
    cacheDict = topo.cacheMemoryDict
    contentGenerator = topo.contentGenerator
    traffic = 0
    hit, hit1, miss = 0,0,0
    runReqDict = {}
    for client in graph.nodes:
        runReqDict[client] = contentGenerator.randomGen(runReqNums)
    traffic = runWithShortestPath(graph, cacheDict,  runReqDict, topo.mainServerId)
    
    return traffic
    
    
class CDNOptimizationProblem(MultiObjectiveTestProblem):
    def __init__(self, topo, runReqNums, tkwargs, batch_size, bounds=[1, 10]):
        self.topo = topo
        self.runReqNums = runReqNums
        self.tkwargs = tkwargs
        self.dim = len(topo.graph.nodes)
        self._bounds = [bounds] * self.dim
        self._ref_point =  self.evaluate_true(batch_size * [[bounds[-1]] * self.dim])[-1]
        self.num_objectives = 2
        super().__init__(noise_std=False, negate=False)
        
    def evaluate_true(self, X):
        traffic = self.traffic_function(X)
        cost =  self.cost_function(X)
        return torch.stack([-1 * traffic, -1 * cost], dim=-1).to(**self.tkwargs)

    # def performance_function(self, x):
    #     results = []
    #     for idx, x_i in enumerate(x):
    #         self.topo.reconfig(x_i, idx)
    #         traffic = runSimulationWithPredefinedDistribution(self.topo, self.runReqNums, idx)
    #         results.append(traffic)
    #     return torch.tensor(results)

        
    def traffic_function(self, x):
        for i in range(len(x)):
            with open("./tmp/save_" + str(i), "wb") as f:
                save_data = [self.topo, self.runReqNums, x[i]]
                pickle.dump(save_data, f)
                
        pool = mp.Pool(processes=NUM_PROCESSORS)
        results = pool.map(self.process_compute_perforamnce, range(len(x)))
        
        return torch.tensor(results)
    
    def process_compute_perforamnce(self, idx):
        with open("./tmp/save_" + str(idx), "rb") as f:
            data = pickle.load(f)
        topo, runReqNums, x = data
        topo.reconfig(x, idx)
        traffic = runSimulationWithPredefinedDistribution(topo, runReqNums, idx)
        return int(traffic)
    
    def cost_function(self, x):
        result = []
        for i in range(len(x)):
            result.append(int(sum(x[i])))
        return torch.tensor(result)