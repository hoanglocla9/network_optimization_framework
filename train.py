import os
import torch
from  problem.network_optimization.topology import NetTopology
from problem.network_optimization.problem import *
import os
import torch

from botorch.models.gp_regression import FixedNoiseGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.sampling import draw_sobol_samples
from botorch.optim.optimize import optimize_acqf, optimize_acqf_list
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement, qNoisyExpectedHypervolumeImprovement
from botorch.utils.sampling import sample_simplex
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement

from botorch import fit_gpytorch_model
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
import numpy as np
from matplotlib import pyplot as plt

import time
import warnings, tqdm
warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

BATCH_SIZE = 32
tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cpu") # "cuda:0" if torch.cuda.is_available() else 
}
SMOKE_TEST = os.environ.get("SMOKE_TEST")
NUM_RESTARTS = 20 if not SMOKE_TEST else 2
RAW_SAMPLES = 1024 if not SMOKE_TEST else 4

N_TRIALS = 3 if not SMOKE_TEST else 2
N_BATCH = 30 if not SMOKE_TEST else 10
MC_SAMPLES = 128  if not SMOKE_TEST else 16

verbose = True

# from botorch.test_functions.multi_objective import DTLZ1, BraninCurrin
# problem = DTLZ3(dim=100, negate=True).to(**tkwargs) # BraninCurrin(negate=True).to(**tkwargs) # 

tempCacheInfo = {'Sydney1': {'size': 100, 'type': 'LRU'}, 'Brisbane2': {'size': 100, 'type': 'LRU'}, 'Canberra1': {'size': 100, 'type': 'LRU'}, \
                 'Sydney2': {'size': 100, 'type': 'LRU'}, 'Townsville': {'size': 100, 'type': 'LRU'}, 'Cairns': {'size': 100, 'type': 'LRU'}, \
                 'Brisbane1': {'size': 100, 'type': 'LRU'}, 'Rockhampton': {'size': 100, 'type': 'LRU'}, 'Armidale': {'size': 100, 'type': 'LRU'}, \
                 'Hobart': {'size': 100, 'type': 'LRU'}, 'Canberra2': {'size': 100, 'type': 'LRU'}, 'Perth1': {'size': 100, 'type': 'LRU'}, \
                 'Perth2': {'size': 100, 'type': 'LRU'}, 'Adelaide1': {'size': 100, 'type': 'LRU'}, 'Adelaide2': {'size': 100, 'type': 'LRU'}, \
                 'Melbourne1': {'size': 100, 'type': 'LRU'}, 'Melbourne2': {'size': 100, 'type': 'LRU'}, 'Alice Springs': {'size': 100, 'type': 'LRU'},\
                 'Darwin': {'size': 100, 'type': 'LRU'}}

bounds = [1, 100]
topology = NetTopology('topology/Aarnet.gml', 'Sydney1', cacheDictInfo=tempCacheInfo)
problem = CDNOptimizationProblem(topology, batch_size=BATCH_SIZE, runReqNums=1000, tkwargs=tkwargs, bounds=bounds)
d=problem.dim
standard_bounds = torch.zeros(2, problem.dim, **tkwargs)
standard_bounds[1] = 1

NOISE_SE = torch.tensor([15.19, 0.63], **tkwargs)

hvs_qparego_all, hvs_qehvi_all, hvs_qnehvi_all, hvs_random_all, hvs_vae_qehvi_all = [], [], [], [], []

def generate_initial_data(n=6):
    # generate training data
    train_x = draw_sobol_samples(
        bounds=problem.bounds,n=1, q=n, seed=torch.randint(1000000, (1,)).item()
    ).squeeze(0).to(**tkwargs)
    train_obj_true = problem(train_x)
    train_obj = train_obj_true + torch.randn_like(train_obj_true) * NOISE_SE
    return train_x, train_obj, train_obj_true


def initialize_model(train_x, train_obj, state_dict=None):
    # define models for objective and constraint
    train_x = normalize(train_x, problem.bounds)
    models = []
    for i in range(train_obj.shape[-1]):
        train_y = train_obj[..., i:i+1]
        train_yvar = torch.full_like(train_y, NOISE_SE[i] ** 2)
        models.append(
            FixedNoiseGP(train_x, train_y, train_yvar, outcome_transform=Standardize(m=1))
        )
        
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return mll, model

def optimize_qehvi_and_get_observation(model, train_x, train_obj, sampler):
    """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
    # partition non-dominated space into disjoint rectangles
    with torch.no_grad():
        pred = model.posterior(normalize(train_x, problem.bounds)).mean
    partitioning = FastNondominatedPartitioning(
        ref_point=problem.ref_point, 
        Y=pred,
    )
    acq_func = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=problem.ref_point,
        partitioning=partitioning,
        sampler=sampler,
    )
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=standard_bounds,
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True,
    )
    # observe new values 
    new_x =  unnormalize(candidates.detach(), bounds=problem.bounds)
    new_obj_true = problem(new_x)
    new_obj = new_obj_true + torch.randn_like(new_obj_true) * NOISE_SE
    return new_x, new_obj, new_obj_true

def optimize_qnehvi_and_get_observation(model, train_x, train_obj, sampler):
    """Optimizes the qEHVI acquisition function, and returns a new candidate and observation."""
    # partition non-dominated space into disjoint rectangles
    acq_func = qNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=problem.ref_point.tolist(),  # use known reference point 
        X_baseline=normalize(train_x, problem.bounds),
        prune_baseline=True,  # prune baseline points that have estimated zero probability of being Pareto optimal
        sampler=sampler,
    )
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=standard_bounds,
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True,
    )
    # observe new values 
    new_x =  unnormalize(candidates.detach(), bounds=problem.bounds)
    new_obj_true = problem(new_x)
    new_obj = new_obj_true + torch.randn_like(new_obj_true) * NOISE_SE
    return new_x, new_obj, new_obj_true


def optimize_qnparego_and_get_observation(model, train_x, train_obj, sampler):
    """Samples a set of random weights for each candidate in the batch, performs sequential greedy optimization 
    of the qNParEGO acquisition function, and returns a new candidate and observation."""
    train_x = normalize(train_x, problem.bounds)
    with torch.no_grad():
        pred = model.posterior(train_x).mean
    acq_func_list = []
    for _ in range(BATCH_SIZE):
        weights = sample_simplex(problem.num_objectives, **tkwargs).squeeze()
        objective = GenericMCObjective(get_chebyshev_scalarization(weights=weights, Y=pred))
        acq_func = qNoisyExpectedImprovement(  # pyre-ignore: [28]
            model=model,
            objective=objective,
            X_baseline=train_x,
            sampler=sampler,
            prune_baseline=True,
        )
        acq_func_list.append(acq_func)
    # optimize
    candidates, _ = optimize_acqf_list(
        acq_function_list=acq_func_list,
        bounds=standard_bounds,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
    )
    # observe new values 
    new_x =  unnormalize(candidates.detach(), bounds=problem.bounds)
    new_obj_true = problem(new_x)
    new_obj = new_obj_true + torch.randn_like(new_obj_true) * NOISE_SE
    return new_x, new_obj, new_obj_true



if __name__ == "__main__":
    # average over multiple trials
    for trial in range(1, N_TRIALS + 1):
        torch.manual_seed(trial)

        print(f"\nTrial {trial:>2} of {N_TRIALS} ", end="")
        hvs_qparego, hvs_qehvi, hvs_qnehvi, hvs_random, hvs_vae_qehvi = [], [], [], [], []

        # call helper functions to generate initial training data and initialize model
        train_x_qparego, train_obj_qparego, train_obj_true_qparego = generate_initial_data(n=2*(problem.dim+1))
        mll_qparego, model_qparego = initialize_model(train_x_qparego, train_obj_qparego)

        train_x_qehvi, train_obj_qehvi, train_obj_true_qehvi = train_x_qparego, train_obj_qparego, train_obj_true_qparego
        train_x_qnehvi, train_obj_qnehvi, train_obj_true_qnehvi = train_x_qparego, train_obj_qparego, train_obj_true_qparego
        # train_x_vae_qehvi, train_obj_vae_qehvi, train_obj_true_vae_qehvi = train_x_qparego, train_obj_qparego, train_obj_true_qparego
        train_x_random, train_obj_random, train_obj_true_random = train_x_qparego, train_obj_qparego, train_obj_true_qparego
        mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi)
        mll_qnehvi, model_qnehvi = initialize_model(train_x_qnehvi, train_obj_qnehvi)
        # mll_vae_qehvi, model_vae_qehvi = initialize_model(train_x_vae_qehvi, train_obj_vae_qehvi)
        # compute hypervolume
        bd = DominatedPartitioning(ref_point=problem.ref_point, Y=train_obj_true_qparego)
        volume = bd.compute_hypervolume().item()

        hvs_qparego.append(volume)
        hvs_qehvi.append(volume)
        hvs_qnehvi.append(volume)
        hvs_random.append(volume)
        # hvs_vae_qehvi.append(volume)
        state_dict = None
        # run N_BATCH rounds of BayesOpt after the initial random batch
        for iteration in tqdm.tqdm(range(1, N_BATCH + 1)):    

            t0 = time.time()
            # fit the models
            fit_gpytorch_model(mll_qparego)
            fit_gpytorch_model(mll_qehvi)
            fit_gpytorch_model(mll_qnehvi)
            # fit_gpytorch_model(mll_vae_qehvi)

            # define the qEI and qNEI acquisition modules using a QMC sampler
            qparego_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)
            qehvi_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)
            qnehvi_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)
            # vae_qehvi_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)

            # optimize acquisition functions and get new observations
            new_x_qparego, new_obj_qparego, new_obj_true_qparego = optimize_qnparego_and_get_observation(
                model_qparego, train_x_qparego, train_obj_qparego, qparego_sampler
            )
            new_x_qehvi, new_obj_qehvi, new_obj_true_qehvi = optimize_qehvi_and_get_observation(
                model_qehvi, train_x_qehvi, train_obj_qehvi, qehvi_sampler
            )
            new_x_qnehvi, new_obj_qnehvi, new_obj_true_qnehvi = optimize_qnehvi_and_get_observation(
                model_qnehvi, train_x_qnehvi, train_obj_qnehvi, qnehvi_sampler
            )
            # new_x_vae_qehvi, new_obj_vae_qehvi, new_obj_true_vae_qehvi = optimize_vae_qehvi_and_get_observation(
            #     model_vae_qehvi, train_x_vae_qehvi, train_obj_vae_qehvi, vae_qehvi_sampler
            # )
            new_x_random, new_obj_random, new_obj_true_random = generate_initial_data(n=BATCH_SIZE)

            # update training points
            train_x_qparego = torch.cat([train_x_qparego, new_x_qparego])
            train_obj_qparego = torch.cat([train_obj_qparego, new_obj_qparego])
            train_obj_true_qparego = torch.cat([train_obj_true_qparego, new_obj_true_qparego])

            train_x_qehvi = torch.cat([train_x_qehvi, new_x_qehvi])
            train_obj_qehvi = torch.cat([train_obj_qehvi, new_obj_qehvi])
            train_obj_true_qehvi = torch.cat([train_obj_true_qehvi, new_obj_true_qehvi])

            train_x_qnehvi = torch.cat([train_x_qnehvi, new_x_qnehvi])
            train_obj_qnehvi = torch.cat([train_obj_qnehvi, new_obj_qnehvi])
            train_obj_true_qnehvi = torch.cat([train_obj_true_qnehvi, new_obj_true_qnehvi])

            # train_x_vae_qehvi = torch.cat([train_x_vae_qehvi, new_x_vae_qehvi])
            # train_obj_vae_qehvi = torch.cat([train_obj_vae_qehvi, new_obj_vae_qehvi])
            # train_obj_true_vae_qehvi = torch.cat([train_obj_true_vae_qehvi, new_obj_true_vae_qehvi])

            train_x_random = torch.cat([train_x_random, new_x_random])
            train_obj_random = torch.cat([train_obj_random, new_obj_random])
            train_obj_true_random = torch.cat([train_obj_true_random, new_obj_true_random])

            # update progress
            for hvs_list, train_obj in zip(
                (hvs_random, hvs_qparego, hvs_qehvi, hvs_qnehvi, hvs_vae_qehvi), 
                (train_obj_true_random, train_obj_true_qparego, train_obj_true_qehvi, train_obj_true_qnehvi), # train_obj_true_vae_qehvi
            ):
                # compute hypervolume
                bd = DominatedPartitioning(ref_point=problem.ref_point, Y=train_obj)
                volume = bd.compute_hypervolume().item()
                hvs_list.append(volume)


            # state_dict = model_vae_qehvi.state_dict()
            # reinitialize the models so they are ready for fitting on next iteration
            # Note: we find improved performance from not warm starting the model hyperparameters
            # using the hyperparameters from the previous iteration
            mll_qparego, model_qparego = initialize_model(train_x_qparego, train_obj_qparego)
            mll_qehvi, model_qehvi = initialize_model(train_x_qehvi, train_obj_qehvi)
            mll_qnehvi, model_qnehvi = initialize_model(train_x_qnehvi, train_obj_qnehvi)
            # mll_vae_qehvi, model_vae_qehvi = initialize_model(train_x_vae_qehvi, train_obj_vae_qehvi)

            t1 = time.time()

            if verbose:
                print(
                    f"\nBatch {iteration:>2}: Hypervolume (random, qNParEGO, qEHVI, qNEHVI) = " # , VAE_qEHVI
                    f"({hvs_random[-1]:>4.2f}, {hvs_qparego[-1]:>4.2f}, {hvs_qehvi[-1]:>4.2f}, {hvs_qnehvi[-1]:>4.2f}), " # , {hvs_vae_qehvi[-1]:>4.2f}
                    f"time = {t1-t0:>4.2f}.", end=""
                )
            else:
                print(".", end="")

        hvs_qparego_all.append(hvs_qparego)
        hvs_qehvi_all.append(hvs_qehvi)
        hvs_qnehvi_all.append(hvs_qnehvi)
        hvs_random_all.append(hvs_random)
        # hvs_vae_qehvi_all.append(hvs_vae_qehvi)
        
        
    def ci(y):
        return 1.96 * y.std(axis=0) / np.sqrt(N_TRIALS)


    iters = np.arange(N_BATCH + 1) * BATCH_SIZE
    log_hv_difference_qparego = problem.max_hv - np.asarray(hvs_qparego_all) # np.log10(problem.max_hv - np.asarray(hvs_qparego_all))
    log_hv_difference_qehvi = problem.max_hv - np.asarray(hvs_qehvi_all) #np.log10(problem.max_hv - np.asarray(hvs_qehvi_all))
    log_hv_difference_qnehvi = problem.max_hv - np.asarray(hvs_qnehvi_all) #np.log10(problem.max_hv - np.asarray(hvs_qnehvi_all))
    log_hv_difference_rnd = problem.max_hv - np.asarray(hvs_random_all) #np.log10(problem.max_hv - np.asarray(hvs_random_all))

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.errorbar(
        iters, log_hv_difference_rnd.mean(axis=0), yerr=ci(log_hv_difference_rnd),
        label="Sobol", linewidth=1.5,
    )
    ax.errorbar(
        iters, log_hv_difference_qparego.mean(axis=0), yerr=ci(log_hv_difference_qparego),
        label="qNParEGO", linewidth=1.5,
    )
    ax.errorbar(
        iters, log_hv_difference_qehvi.mean(axis=0), yerr=ci(log_hv_difference_qehvi),
        label="qEHVI", linewidth=1.5,
    )
    ax.errorbar(
        iters, log_hv_difference_qnehvi.mean(axis=0), yerr=ci(log_hv_difference_qnehvi),
        label="qNEHVI", linewidth=1.5,
    )
    ax.set(xlabel='number of observations (beyond initial points)', ylabel='Log Hypervolume Difference')
    ax.legend(loc="lower left")
    
    fig.savefig('foo.png')
