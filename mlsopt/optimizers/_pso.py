from joblib import delayed, Parallel
import logging
from math import exp
import numpy as np
import pandas as pd
import time
from typing import Optional, Tuple, Union

# Package imports
from ..base import BaseOptimizer
from ..constants import I_DISTRIBUTIONS, F_DISTRIBUTIONS, FEATURE_SAMPLER, HP_SAMPLER, T_NUMERIC
from ..utils import pandas_cast_dtypes, pandas_safe_iloc, sigmoid

_LOGGER = logging.getLogger(__name__)


class PSOptimizer(BaseOptimizer):
    """Particle swarm optimization (PSO) that uses a binary PSO algorithm for 
    features and a continuous PSO algorithm for hyperparameters.

    Parameters
    ----------
    n_particles : int
        Number of particles in swarm.
        
    omega_bounds : tuple, optional (default=(0.10, 0.90))
        ADD HERE.
    
    v_bounds : tuple or None, optional (default=None)
        ADD HERE.    
    """
    def __init__(self, 
                 n_particles: int,
                 omega_bounds: Tuple[T_NUMERIC, T_NUMERIC] = (0.10, 0.90),
                 v_bounds: Optional[Tuple[T_NUMERIC, T_NUMERIC]] = None,
                 phi_p: T_NUMERIC = 1.0,
                 phi_g: T_NUMERIC = 2.0,
                 max_iterations: int = 10,
                 tolerance: float = 1e-6,
                 n_jobs: int = 1,
                 backend: str = 'loky',
                 verbose: Union[bool, int] = 0,
                 seed: Optional[int] = None) -> None:
        self.n_particles      = n_particles
        self.n_configurations = n_particles  # Used in base class
        self.omega_bounds     = omega_bounds
        self.v_bounds         = v_bounds
        self.phi_p            = phi_p
        self.phi_g            = phi_g
        self.max_iterations   = max_iterations
        self.tolerance        = tolerance

        super().__init__(backend=backend,
                         n_jobs=n_jobs,
                         verbose=verbose,
                         seed=seed)   

    def _evaluate(self, 
                  objective, 
                  iteration):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        # Package particles into list of dictionaries for objective function
        particles = []
        for i in range(self.n_configurations):
            particle = {}
            for sname in self._swarm.keys():
                x = self._swarm[sname]['x']
                if self._swarm[sname]['algo'] == 'cpso': 
                    # Hack to handle conversion of integers to float when to_dict
                    # https://stackoverflow.com/questions/31374928/dtype-changes-when-using-dataframe-to-dict
                    particle[sname] = pandas_safe_iloc(x, iloc=i).to_dict() 
                else:
                    particle[sname] = x.iloc[i].to_list()
            particles.append(particle)
        
        # Calculate metrics for particles
        results = Parallel(n_jobs=self.n_jobs, verbose=False, backend=self.backend)\
                        (delayed(self._evaluate_single_config)
                            (objective, particle, i, iteration)
                                for i, particle in enumerate(particles))
        
        # Sort results based on metric
        sorted_results = sorted(results, 
                                key=lambda x: x['metric'], 
                                reverse=~self.lower_is_better)
        
        # Add to history
        self.history.append(sorted_results)

        # Get metrics
        obj = np.array([r['metric'] for r in results])
        
        return obj
    
    def _initialize_swarm(self, objective, sampler):
        """ADD HERE
        
        Parameters
        ----------
        
        Returns
        -------
        """
        # Sample space (keep separate for linear algebra computations during updates)
        x = sampler.sample_space(n_samples=self.n_particles, 
                                 return_combined=False)

        # Initialize data for each sampler
        worst_obj   = np.inf if self.lower_is_better else -np.inf
        self._swarm = {}
        self._best  = {'gbest_o': worst_obj}
        for sname, sdist in sampler._registered.items():
            meta = {}
            
            # Add particles
            meta['x'] = x.pop(sname)
            
            # Add other metadata
            if sdist.__type__ == FEATURE_SAMPLER:
                meta['pnames'] = sdist.feature_names
                meta['algo']   = 'bpso'
                lb             = [0] * len(meta['pnames']) 
                ub             = [1] * len(meta['pnames'])
            elif sdist.__type__ == HP_SAMPLER:
                dtypes = {}
                pnames = []
                lb     = []
                ub     = []
                for hp in sdist.space.get_hyperparameters():
                    # Define data types
                    dist = hp.__class__.__name__
                    if dist in I_DISTRIBUTIONS:
                        dtypes[hp.name] = int
                    elif dist in F_DISTRIBUTIONS:
                        dtypes[hp.name] = float
                    else:
                        _LOGGER.warn(f"distribution of hyperparameter <{hp.name}> " + 
                                     f"is {dist} and not supported, skipping")
                        continue                        
                    pnames.append(hp.name)
                    lb.append(hp.lower)
                    ub.append(hp.upper)

                meta['pnames'] = np.array(pnames)
                meta['algo']   = 'cpso'
                meta['dtypes'] = dtypes
            else:
                msg = f"sampler type {sdist.__type__} not recognized"
                _LOGGER.error(msg)
                raise ValueError(msg)
            
            # Update dictionary
            meta['dimensions'] = len(meta['pnames'])
            meta['pbest_x']    = meta['x'].copy()
            meta['pbest_o']    = np.array([worst_obj] * self.n_configurations)
            
            # Initialize velocities
            meta['lb'] = np.array(lb)
            meta['ub'] = np.array(ub)
            diff      = meta['ub'] - meta['lb']
            meta['v'] = np.column_stack(
                [self.rng.uniform(-diff[i], diff[i], size=self.n_particles) 
                    for i in range(meta['dimensions'])]
            )            
            # Update dictionaries
            self._swarm[sname] = meta
            self._best[sname]  = {'gbest_x': {}}
            
    def _update_particles(self, iteration: int) -> None:
        """Update particles' positions and velocities.
        
        Parameters
        ----------
        iteration : int
            ADD HERE.
        
        Returns
        -------
        None
        """
        if self.verbose:
            _LOGGER.info("updating particles' positions and velocities")
    
        # Calculate new omega
        diff  = self.omega_bounds[1] - self.omega_bounds[0]
        omega = self.omega_bounds[1] - (iteration - 2) * (diff / self.max_iterations)
    
        for sname in self._swarm.keys():
            # Random numbers
            meta = self._swarm[sname]
            size = (self.n_configurations, meta['dimensions'])
            rp   = self.rng.uniform(size=size)
            rg   = self.rng.uniform(size=size)

            # Update velocities
            v       = meta.pop('v')
            x       = meta.pop('x').astype(float)
            pbest_x = meta['pbest_x'].astype(float)
            gbest_x = self._best[sname]['gbest_x'].astype(float)
            v       = omega * v + \
                        self.phi_p * rp * (pbest_x - x) + \
                        self.phi_g * rg * (gbest_x - x)
            
            # Adjust particles' velocities based on bounds
            if self.v_bounds is not None:
                mask_lb = v < self.v_bounds[0]
                mask_ub = v > self.v_bounds[1]
                v       = v * (~np.logical_or(mask_lb, mask_ub)) + \
                            self.v_bounds[0] * mask_lb + self.v_bounds[1] * mask_ub
            
            # Algorithm specific updates for CPSO vs BPSO
            if meta['algo'] == 'cpso':
                # Update particles' positions and adjust based on bounds
                x       += v
                mask_lb  = x < meta['lb']
                mask_ub  = x > meta['ub']
                x        = x * (~np.logical_or(mask_lb, mask_ub)) + \
                            meta['lb'] * mask_lb + meta['ub'] * mask_ub
                
                # Cast data types now
                x = pandas_cast_dtypes(x, dtypes=meta['dtypes'])
                
            else:
                # Update particles positions using sigmoid transfer function
                s = sigmoid(v)
                r = self.rng.uniform(size=size)
                x = (r < s).astype(bool)
            
            # Update dictionaries
            meta['v']          = v
            meta['x']          = x
            self._swarm[sname] = meta   
    
    def _update_best_results(self, obj):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        if self.verbose:
            _LOGGER.info("updating best results for particles and swarm")

        p_idx     = None
        g_idx     = None
        improve   = None
        converged = None
        for sname in self._swarm.keys():
            meta = self._swarm[sname]
            if p_idx is None:
                p_idx = (obj < meta['pbest_o']) if self.lower_is_better else \
                          (obj > meta['pbest_o'])

            # Update particles' best positions and objective values
            meta['pbest_x'][p_idx] = meta['x'][p_idx].copy()
            meta['pbest_o'][p_idx] = obj[p_idx].copy()
            
            # Update swarm's best position and objective value and check for convergence
            if g_idx is None:
                g_idx = np.argmin(meta['pbest_o']) if self.lower_is_better else \
                          np.argmax(meta['pbest_o'])   
            if improve is None:
                improve = meta['pbest_o'][g_idx] < self._best['gbest_o'] if self.lower_is_better \
                            else meta['pbest_o'][g_idx] > self._best['gbest_o']
            if improve:
                self._best[sname]['gbest_x'] = pandas_safe_iloc(meta['x'], iloc=g_idx)
      
                # Check for convergence of search
                if converged is None:
                    diff      = np.linalg.norm(self._best['gbest_o'] - meta['pbest_o'][g_idx])
                    converged = diff < self.tolerance
                
                # Update swarm best objective 
                self._best['gbest_o'] = meta['pbest_o'][g_idx].copy()

        return converged

    def search(self, 
               objective, 
               sampler, 
               lower_is_better):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        tic = time.time()
        
        # Initialize parameters
        self._initialize(sampler=sampler, lower_is_better=lower_is_better)

        # Initialize swarm
        if self.verbose: 
            _LOGGER.info("initializing swarm")
        self._initialize_swarm(objective, sampler)
        
        # Continue search
        iteration = 1
        while iteration <= self.max_iterations:
            if self.verbose:
                msg = "\n" + "*"*40 + f"\niteration {iteration}\n" + "*"*40
                _LOGGER.info(msg)

            # 1. Evaluate particles
            obj = self._evaluate(objective=objective, iteration=iteration)
            
            # 2. Update best results
            converged = self._update_best_results(obj)
            
            # If converged, early stopping
            if converged:
                if self.verbose:
                    _LOGGER.info("optimization converged: " + 
                                 f"{iteration}/{self.max_iterations} - stopping " + 
                                 "criteria below tolerance")
                break

            # 3. Update particles         
            self._update_particles(iteration=iteration)
            
            # Keep searching
            iteration += 1

        # Clean up data
        del self._swarm, self._best

        # Finished searching
        toc = time.time()
        minutes = round((toc - tic) / 60, 2)
        if self.verbose:
            _LOGGER.info(f"finished searching in {minutes} minutes") 
        
        return self._optimal_solution()