import autograd.numpy as np
import pickle
from .problem import Problem
from pymoo.util.normalization import normalize
import multiprocessing as mp
NUM_PROCESSORS = 4
import random, os, re
from problem.network_optimization.util.virtual_run import runWithShortestPath

def runSimulationWithPredefinedDistribution(topo, runReqNums):
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

class CDN(Problem):
    ## 324, 5188
    def __init__ (self, n_var=4, n_obj=2, n_constr=0, xl=10, xu=180, min_cost=0.5, transformToInteger=False):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu)
        self.min_cost = min_cost
        self.count_step = 0
        self.transformToInteger = transformToInteger
        
        
    def compute_y_bounds(self):
        performance_lower = 1.0 * self.traffic_function([self.xu])
        cost_lower = self.cost_function([self.xl])
        performance_upper = 1.0 * self.traffic_function([self.xl])
        cost_upper = self.cost_function([self.xu])
        
        self.performance_bounds = [performance_lower, performance_upper]
        self.cost_bounds = [cost_lower, cost_upper]
        
    
class CDN_PLACEMENT(CDN):
    def __init__(self, n_var=4, n_obj=2, n_constr=0, xl=0, xu=1, min_cost=0.5, transformToInteger=False):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, xl=xl, xu=xu, min_cost=min_cost, transformToInteger=transformToInteger)
        
    def _calc_pareto_front(self, n_pareto_points=100):
        raise "Not implement yet"
    
    def set_parameters(self, topo, runReqNums):
        self.topo = topo
        self.runReqNums = runReqNums
        self.compute_y_bounds()
        
    def _evaluate(self, x, out,  *args, **kwargs):
        x_temp = np.round(x.copy())
        self.count_step += len(x_temp)
        performance =  self.traffic_function(x_temp)
        cost = self.cost_function(x_temp)
        
        del x, x_temp
        normalized_performance = (performance - np.ones(performance.shape) * self.performance_bounds[0]) / (self.performance_bounds[1] - self.performance_bounds[0])
        normalized_cost = (cost - self.cost_bounds[0]) / (self.cost_bounds[1] - self.cost_bounds[0])
        out["F"] = np.column_stack([normalized_performance, normalized_cost])
                    
    def traffic_function(self, x):
        dataList = []
        for i in range(len(x)):
            dataList.append([self.topo, self.runReqNums, x[i]]) 
        pool = mp.Pool(processes=NUM_PROCESSORS)
        results = pool.map(self.process_compute_perforamnce, dataList)
        
        return np.array(results)
    
    def process_compute_perforamnce(self, data):
        topo, runReqNums, x = data
        topo.reconfig(x)
        traffic = runSimulationWithPredefinedDistribution(topo, runReqNums)
        return int(traffic)
    
    def cost_function(self, x):
        result = []
        for i in range(len(x)):
            result.append(int(sum(x[i])))
        return np.array(result)
    
    
class CDN_RAM(CDN):
    def _calc_pareto_front(self, n_pareto_points=100):
        raise "Not implement yet"
        
    def set_parameters(self, topo, runReqNums):
        self.topo = topo
        self.runReqNums = runReqNums
        self.compute_y_bounds()
        
    def _evaluate(self, x, out,  *args, **kwargs):
        x_temp = np.round(x.copy())
        self.count_step += len(x_temp)
        performance =  self.traffic_function(x_temp)
        cost = self.cost_function(x_temp)
        del x, x_temp
        normalized_performance = (performance - np.ones(performance.shape) * self.performance_bounds[0]) / (self.performance_bounds[1] - self.performance_bounds[0])
        normalized_cost = (cost - self.cost_bounds[0]) / (self.cost_bounds[1] - self.cost_bounds[0])
        out["F"] = np.column_stack([normalized_performance, normalized_cost])
#         out["G"] = int(self.min_cost * (80*1024-10*1024) + 10 *1024) - cost
                    
    def traffic_function(self, x):
        for i in range(len(x)):
            with open("./tmp/save_" + str(i), "wb") as f:
                save_data = [self.topo, self.runReqNums, x[i]]
                pickle.dump(save_data, f)
        pool = mp.Pool(processes=NUM_PROCESSORS)
        results = pool.map(self.process_compute_perforamnce, range(len(x)))
        
        return np.array(results)
    
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
        return np.array(result)
