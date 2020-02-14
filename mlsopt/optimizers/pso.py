from joblib import delayed, Parallel
import logging
from math import exp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import time

matplotlib.style.use("ggplot")

# Package imports
from ..base.optimizers import BaseOptimizer
from ..utils import parse_hyperopt_param, STATUS_FAIL, STATUS_OK

__all__ = ["PSOptimizer"]
_LOGGER = logging.getLogger(__name__)


class PSOptimizer(BaseOptimizer):
    """ADD HERE.

    Parameters
    ----------
    """
    def __init__(self, 
                 n_particles,
                 omega_bounds=(0.10, 0.90),
                 v_bounds=None,
                 phi_p=1.0,
                 phi_g=2.0,
                 max_iterations=10,
                 tolerance=1e-6,
                 n_jobs=1,
                 backend='loky',
                 verbose=0,
                 seed=None):
        self.n_particles    = n_particles
        self.omega_bounds   = omega_bounds
        self.v_bounds       = v_bounds
        self.phi_p          = phi_p
        self.phi_g          = phi_g
        self.max_iterations = max_iterations
        self.tolerance      = tolerance

        super().__init__(backend=backend,
                         n_jobs=n_jobs,
                         verbose=verbose,
                         seed=seed)   
    
    def __str__(self):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        pass

    def __repr__(self):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        pass

    def _sigmoid(self, x):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        return 1 / (1 + np.exp(-x))

    def _initialize_swarm(self, sampler):
        """Initialize the swarm.
        
        Parameters
        ----------
        
        Returns
        -------
        """
        # Binary PSO
        b_lb         = []
        b_ub         = []
        b_pnames     = []
        b_sname      = None
        b_dimensions = 0

        # Continuous PSO
        c_lb           = []
        c_ub           = []
        c_pnames       = []
        c_sname        = None
        c_dimensions   = 0
        self._c_dtypes = {}

        # Multiplier
        max_score = np.inf if self.lower_is_better else -np.inf

        for sname, sdist in sampler.samplers.items():
            if sdist.__type__ == 'feature':
                pnames        = sdist.space.keys()
                n_dimensions  = len(pnames)
                b_lb          = [0] * n_dimensions
                b_ub          = [1] * n_dimensions 
                b_dimensions  = n_dimensions
                b_pnames      = pnames
                b_sname       = sname
            else:
                c_sname = sname
                for pname, pdist in sdist.space.items():
                    dname, bounds = parse_hyperopt_param(str(pdist))
                    if dname in ['choice', 'pchoice']: continue
                    
                    lb, ub = bounds[:2]
                    if 'log' in dname:
                        lb, ub = exp(lb), exp(ub)
                
                    c_lb.append(lb)
                    c_ub.append(ub)
                    c_dimensions += 1
                    c_pnames.append(pname)

                    # Define dtypes
                    if dname.startswith("q"):
                        self._c_dtypes[pname] = int
                    else:
                        self._c_dtypes[pname] = float
        
        # Convert bounds to numpy arrays
        b_lb = np.array(b_lb)
        b_ub = np.array(b_ub)
        c_lb = np.array(c_lb)
        c_ub = np.array(c_ub)

        rv = self.rg.uniform

        # Initialize binary particle positions and velocities
        diff = b_ub - b_lb
        b_x  = np.round(rv(size=(self.n_particles, b_dimensions)))
        b_v  = np.column_stack(
            [rv(-diff[i], diff[i], size=self.n_particles) for i in range(b_dimensions)]
            )
    
        # Initialize continuous particle positions and velocities
        pts  = rv(size=(self.n_particles, c_dimensions))
        diff = c_ub - c_lb
        c_x  = c_lb + pts * (c_ub - c_lb)
        c_v  = np.column_stack(
            [rv(-diff[i], diff[i], size=self.n_particles) for i in range(c_dimensions)]
            )
        
        # Initialize particle best and swarm best results
        pbest_x = np.zeros((self.n_particles, b_dimensions + c_dimensions))
        pbest_o = np.repeat(max_score, self.n_particles)
        gbest_x = np.zeros(b_dimensions + c_dimensions)
        gbest_o = max_score

        return {
            'b_lb'         : b_lb,
            'b_ub'         : b_ub,
            'b_x'          : b_x,
            'b_v'          : b_v,
            'b_pnames'     : b_pnames,
            'b_sname'      : b_sname,
            'b_dimensions' : b_dimensions,

            'c_lb'         : c_lb,
            'c_ub'         : c_ub,
            'c_x'          : c_x,
            'c_v'          : c_v,
            'c_pnames'     : c_pnames,
            'c_sname'      : c_sname,
            'c_dimensions' : c_dimensions,

            'pbest_x'      : pbest_x,
            'pbest_o'      : pbest_o,
            'gbest_x'      : gbest_x,
            'gbest_o'      : gbest_o
        }
        
    def _calculate_single_particle(self, 
                                   objective, 
                                   particle, 
                                   i, 
                                   iteration):
        """ADD HERE
        
        Parameters
        ----------
        
        Returns
        -------
        """
        if self.verbose and i % self.n_jobs == 0:
            _LOGGER.info(f"evaluating particle, {self.n_particles - i} " + 
                            "remaining")
    
        # Evaluate particle
        results = objective(particle)

        # Check if failure occurred during objective func evaluation
        if STATUS_FAIL in results['status'].upper() or STATUS_OK not in results['status'].upper():
            msg = "running particle failed"
            if 'message' in results.keys():
                if results['message']: msg += f" because {results['message']}"
            _LOGGER.warn(msg)
            return {
                'status'    : results['status'],
                'metric'    : np.inf if self.lower_is_better else -np.inf,
                'params'    : particle,
                'iteration' : iteration,
                'id'        : i
            }
        
        # Find best metric so far and compare results to see if current result is better
        if self.lower_is_better:
            if results['metric'] < self.best_results['metric']:
                if self.verbose:
                    _LOGGER.info(f"new best metric {round(results['metric'], 4)}")
                self.best_results['metric'] = results['metric']
                self.best_results['params'] = particle        
        else:
            if results['metric'] > self.best_results['metric']:
                if self.verbose:
                    _LOGGER.info(f"new best metric {round(results['metric'], 4)}")
                self.best_results['metric'] = results['metric']
                self.best_results['params'] = particle        
 
        return {
            'status'    : results['status'],
            'metric'    : results['metric'],
            'params'    : particle,
            'iteration' : iteration,
            'id'        : i
            }
        
    def _evaluate(self, 
                  objective, 
                  iteration, 
                  *, 
                  b_x=None, 
                  c_x=None,
                  b_pnames=None,
                  c_pnames=None,
                  b_sname=None,
                  c_sname=None):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        # Package into dictionary for objective function
        particles = []
        for i in range(self.n_particles):
            particle = {}
            if len(b_x):
                particle[b_sname] = b_x[i].astype(bool)
            if len(c_x):
                particle[c_sname] = {}
                for c_pname, value in zip(c_pnames, c_x[i]):
                    # Check data type and cast if needed
                    if self._c_dtypes[c_pname] == int: 
                        value = int(np.round(value))
                    particle[c_sname][c_pname] = value
            
            # Add particle to list
            particles.append(particle)

        # Calculate metrics for particles
        results = Parallel(n_jobs=self.n_jobs, verbose=False, backend=self.backend)\
                        (delayed(self._calculate_single_particle)
                            (objective, particle, i, iteration)
                                for i, particle in enumerate(particles))
        
        # Sort results based on metric
        sorted_results = sorted(results, 
                                key=lambda x: x['metric'], 
                                reverse=~self.lower_is_better)
        
        # Add to history
        self.history.append(sorted_results)

        # Return only metrics
        obj = np.array([r['metric'] for r in results])
        return obj

    def _get_best_results(self,
                          pbest_x,
                          gbest_x,
                          b_x,
                          c_x,
                          b_dimensions,
                          c_dimensions):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        # Get best results for binary and/or continuous particles based on case
        b_pbest_x = np.array([])
        b_gbest_x = np.array([])
        c_pbest_x = np.array([])
        c_gbest_x = np.array([])

        # Case 1: Binary and continuous particles
        if b_dimensions and c_dimensions:
            b_pbest_x = pbest_x[:, :b_dimensions].copy()
            b_gbest_x = gbest_x[:b_dimensions].copy()

            c_pbest_x = pbest_x[:, b_dimensions:].copy()
            c_gbest_x = gbest_x[b_dimensions:].copy()

        # Case 2: Only binary particles
        elif b_dimensions and not c_dimensions:
            b_pbest_x = pbest_x.copy()
            b_gbest_x = gbest_x.copy()

        # Case 3: Only continuous particles
        elif not b_dimensions and c_dimensions:
            c_pbest_x = pbest_x.copy()
            c_gbest_x = gbest_x.copy()

        # Case 4: Error
        else:
            _LOGGER.exception("no particles defined")
            raise ValueError

        return {
            'b_pbest_x' : b_pbest_x,
            'b_gbest_x' : b_gbest_x,
            'c_pbest_x' : c_pbest_x,
            'c_gbest_x' : c_gbest_x
        }


    def _update_best_results(self, 
                             o,
                             pbest_x,
                             pbest_o,
                             gbest_x,
                             gbest_o,
                             b_x=None, 
                             c_x=None, 
                             b_dimensions=None, 
                             c_dimensions=None):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        # Update particle best objective values
        p_idx          = (o < pbest_o) if self.lower_is_better else (o > pbest_o)
        pbest_o[p_idx] = o[p_idx]
        
        # Case 1: Binary and continuous particles
        if b_dimensions and c_dimensions:
            pbest_x[p_idx] = np.column_stack([
                b_x[p_idx].copy(), c_x[p_idx].copy()
            ])
        
        # Case 2: Only binary particles
        elif b_dimensions and not c_dimensions:
            pbest_x[p_idx] = b_x[p_idx].copy()

        # Case 3: Only continuous particles
        elif not b_dimensions and c_dimensions:
            pbest_x[p_idx] = c_x[p_idx].copy()

        # Case 4: Error
        else:
            _LOGGER.exception("no particles defined")
            raise ValueError
        
        # Check for new best swarm and convergence
        g_idx     = np.argmin(pbest_o) if self.lower_is_better else np.argmax(pbest_o)
        new_best  = pbest_o[g_idx] < gbest_o if self.lower_is_better \
                        else pbest_o[g_idx] > gbest_o
        converged = False
        if new_best:
            diff = np.linalg.norm(gbest_o - pbest_o[g_idx])
            if diff < self.tolerance:
                converged = True 

            # Update best swarm results based on case
            gbest_o = pbest_o[g_idx]

            # Case 1: Binary and continuous particles
            if b_dimensions and c_dimensions:
                gbest_x = np.concatenate([
                    b_x[g_idx].copy(), c_x[g_idx].copy()
                ])
            
            # Case 2: Only binary particles
            elif b_dimensions and not c_dimensions:
                gbest_x = b_x[g_idx].copy()

            # Case 3: Only continuous particles
            elif not b_dimensions and c_dimensions:
                gbest_x = c_x[g_idx].copy()

            # Case 4: Error
            else:
                _LOGGER.exception("no particles defined")
                raise ValueError
        
        return {
            'pbest_x'   : pbest_x,
            'pbest_o'   : pbest_o,
            'gbest_x'   : gbest_x,
            'gbest_o'   : gbest_o,
            'converged' : converged,
            'new_best'  : new_best
        }

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
        start = time.time()
        
        # Get feature names if specified in sampler
        self._colmapper = {}
        for sname, sdist in sampler.samplers.items():
            if sdist.__type__ == 'feature':
                self._colmapper[sname] = sdist.feature_names

        # Set attributes
        self.lower_is_better = lower_is_better

        if self.verbose:
            _LOGGER.info(f"starting {self.__typename__} with {self.n_jobs} " +
                         f"jobs using {self.backend} backend")

        # Create swarm and set best results
        if self.verbose: _LOGGER.info("initializing swarm")
        params                      = self._initialize_swarm(sampler)
        self.best_results['metric'] = np.inf if lower_is_better else -np.inf
        
        # Get parameters
        b_lb         = params.pop("b_lb")
        b_ub         = params.pop("b_ub")
        b_x          = params.pop("b_x")
        b_v          = params.pop("b_v")
        b_dimensions = params.pop("b_dimensions")
        b_pnames     = params.pop("b_pnames")
        b_sname      = params.pop("b_sname")

        c_lb         = params.pop("c_lb")
        c_ub         = params.pop("c_ub")
        c_x          = params.pop("c_x")
        c_v          = params.pop("c_v")
        c_dimensions = params.pop("c_dimensions")
        c_pnames     = params.pop("c_pnames")
        c_sname      = params.pop("c_sname")

        pbest_x      = params.pop("pbest_x")
        pbest_o      = params.pop("pbest_o")
        gbest_x      = params.pop("gbest_x")
        gbest_o      = params.pop("gbest_o")
        
        iteration = 1
        while iteration <= self.max_iterations:
            
            if self.verbose:
                msg = "\n" + "*"*40 + f"\niteration {iteration}\n" + "*"*40
                _LOGGER.info(msg)

            # Iteratively update omega
            omega = self.omega_bounds[1] - \
                        iteration * ((self.omega_bounds[1] - self.omega_bounds[0]) / self.max_iterations)

            # Get current best results
            best = self._get_best_results(pbest_x=pbest_x,
                                          gbest_x=gbest_x,
                                          b_x=b_x,
                                          c_x=c_x,
                                          b_dimensions=b_dimensions,
                                          c_dimensions=c_dimensions)
        
            b_pbest_x = best.pop("b_pbest_x")
            b_gbest_x = best.pop("b_gbest_x")
            c_pbest_x = best.pop("c_pbest_x")
            c_gbest_x = best.pop("c_gbest_x")

            ####################
            # Binary Particles #
            ####################

            if b_dimensions:
                # Update particle velocities
                size = (self.n_particles, b_dimensions)
                rp   = self.rg.uniform(size=size)
                rg   = self.rg.uniform(size=size)
                b_v  = omega*b_v + \
                        self.phi_p*rp*(b_pbest_x - b_x) + \
                        self.phi_g*rg*(b_gbest_x - b_x)

                # # Adjust particle velocities based on bounds
                if self.v_bounds:
                    mask_lb = b_v < self.v_bounds[0]
                    mask_ub = b_v > self.v_bounds[1]
                    b_v     = b_v*(~np.logical_or(mask_lb, mask_ub)) + \
                                self.v_bounds[0]*mask_lb + self.v_bounds[1]*mask_ub
                    
                # Update particle positions using sigmoid transfer function
                s   = self._sigmoid(b_v)
                r   = self.rg.uniform(size=s.shape)
                b_x = (r < s).astype(int)

            ########################
            # Continuous Particles #
            ########################

            if c_dimensions:
                # Update particle velocities
                size = (self.n_particles, c_dimensions)
                rp   = self.rg.uniform(size=size)
                rg   = self.rg.uniform(size=size)
                c_v  = omega*c_v + \
                        self.phi_p*rp*(c_pbest_x - c_x) + \
                        self.phi_g*rg*(c_gbest_x - c_x)

                # # Adjust particle velocities based on bounds
                if self.v_bounds:
                    mask_lb = c_v < self.v_bounds[0]
                    mask_ub = c_v > self.v_bounds[1]
                    c_v     = c_v*(~np.logical_or(mask_lb, mask_ub)) + \
                                self.v_bounds[0]*mask_lb + self.v_bounds[1]*mask_ub

                # Update particle positions
                c_x += c_v

                # Adjust particle positions based on bounds
                mask_lb  = c_x < c_lb
                mask_ub  = c_x > c_ub
                c_x      = c_x*(~np.logical_or(mask_lb, mask_ub)) + \
                            c_lb*mask_lb + c_ub*mask_ub

            ######################
            # Evaluate Particles #
            ######################

            o = self._evaluate(objective, 
                               iteration,
                               b_x=b_x,
                               c_x=c_x,
                               b_pnames=b_pnames,
                               c_pnames=c_pnames,
                               b_sname=b_sname,
                               c_sname=c_sname)

            # Update best results for particles
            results = self._update_best_results(o=o, 
                                                pbest_x=pbest_x,
                                                pbest_o=pbest_o,
                                                gbest_x=gbest_x,
                                                gbest_o=gbest_o,
                                                b_x=b_x,
                                                c_x=c_x,
                                                b_dimensions=b_dimensions,
                                                c_dimensions=c_dimensions)
            
            pbest_x   = results.pop("pbest_x")
            pbest_o   = results.pop("pbest_o")
            gbest_x   = results.pop("gbest_x")
            gbest_o   = results.pop("gbest_o")
            converged = results.pop("converged")
            new_best  = results.pop("new_best")

            # If converged, early stop
            if converged:
                if self.verbose:
                    _LOGGER.info("optimization converged: " + 
                                f"{iteration}/{self.max_iterations} - stopping " + 
                                "criteria below tolerance")
                break

            if new_best and self.verbose:
                _LOGGER.info("new swarm best: " + 
                            f"{iteration}/{self.max_iterations} - {round(gbest_o, 4)}")

            # Continue optimization
            iteration += 1

        # Finished searching
        minutes = round((time.time() - start) / 60, 2)
        if self.verbose:
            _LOGGER.info(f"finished searching in {minutes} minutes") 
        
        return self

    def serialize(self, save_name):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        if not save_name.endswith(".csv"): save_name += ".csv"
        
        # Concatenate history together
        df = pd.concat([pd.DataFrame(h) for h in self.history], axis=0)\
               .reset_index(drop=True)

        # Unroll parameters
        df_params = df.pop('params')
        for sname in df_params.iloc[0].keys():
            columns = self._colmapper.get(sname, None)
            records = df_params.apply(lambda x: x[sname]).values.tolist()
            df      = pd.concat([df, 
                                 pd.DataFrame.from_records(records, columns=columns)
                                 ], axis=1)

        # Write data to disk
        df.sort_values(by='metric', ascending=self.lower_is_better)\
          .to_csv(save_name, index=False)
        if self.verbose:
            _LOGGER.info(f"saved results to disk at {save_name}")

    def plot_history(self):
        """ADD
        
        Parameters
        ----------
        
        Returns
        -------
        """
        # Concatenate history together
        df = pd.concat([pd.DataFrame(h) for h in self.history], axis=0)\
               .reset_index(drop=True)
        
        # Boxplot and swarmplot
        sns.boxplot(x='iteration', y='metric', data=df)
        sns.swarmplot(x='iteration', y='metric', data=df, color=".25")

        # Decorate plots
        best   = df.iloc[df['metric'].idxmin()] if self.lower_is_better \
                    else df.iloc[df['metric'].idxmax()]
        title  = f"Best metric = {round(best['metric'], 4)} found in " + \
                 f"iteration {best['iteration']}"
        plt.title(title)
        plt.tight_layout()
        plt.show()