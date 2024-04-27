"""
Module Name: noise
Description: This module provides functions for downsampling time series data.
Author: Alexander James
"""

# Import necessary modules
import numpy as np
import pyleoclim as pyleo

def downsample(series,method='exponential',param=[1],return_index=False,seed=None):
    '''Function to downsample a time series by randomly selecting time increments.

    Parameters
    ----------

    series : pyleoclim.core.Series object
        The time series to be downsampled.

    method : str
        the probability distribution of the random index increments.
        possible choices include 'exponential', 'poisson', 'pareto', or 'random_choice'.
        if 'exponential', `param` is expected to be a single scale parameter (traditionally denoted \lambda)
        if 'poisson', `param` is expected to be a single parameter (rate)
        if 'pareto', expects a 2-list with 2 scalar shape & scale parameters (in that order)
        if 'random_choice', expects a 2-list containing the arrays:      
            value_random_choice: 
                elements from which the random sample is generated (e.g. [1,2])
            random_choice: 
                probabilities associated with each entry value_random_choice  (e.g. [.95,.05])
            (These two arrays must be of the same size)
            
    param : list
        The parameter(s) of the chosen distribution. See `delta_t_dist` for details.

    return_index : bool
        If True, the function returns the index of the downsampled time series.
        If False, the function returns the downsampled time series itself

    seed : int
        Seed for the random number generator.
        
    Returns
    -------
    
    downsampled_series : pyleoclim.core.Series object
        The down sampled time series.
        
    Examples
    --------
    
    SOI = pyleo.utils.load_dataset('SOI')
    SOI_downsampled = downsample(SOI,delta_t_dist='exponential',param=[1.0])
    SOI_downsampled.plot()
    '''

    if seed is not None:
        np.random.default_rng(seed)
    
    valid_distributions = ["exponential", "poisson", "pareto", "random_choice"]
    if method not in valid_distributions:
        raise ValueError("delta_t_dist must be one of: 'exponential', 'poisson', 'pareto', 'random_choice'.")    
    
    param = np.array(param)    
    n = len(series.time)

    if method == "exponential":
        # make sure that param is of len 1
        if len(param) != 1:
            raise ValueError('The Exponential law takes a single scale parameter.')       
        delta_t = np.random.exponential(scale = param, size=n)
        
    elif method == "poisson":
        if len(param) != 1:
            raise ValueError('The Poisson law takes a single parameter.')       
        delta_t = np.random.poisson(lam = param, size = n) + 1
    elif method == "pareto":
        if len(param) != 2:
            raise ValueError('The Pareto law takes a shape and a scale parameter (in that order) ')
        else:
            delta_t = (np.random.pareto(param[0], n) + 1) * param[1]
    elif method == "random_choice":
        if len(param)<2 or len(param[0]) != len(param[1]):
            raise ValueError("value_random_choice and prob_random_choice must have the same size.")
        delta_t = np.random.choice(param[0], size=n, p=param[1])

    #create time index
    delta_t_tuned = []
    for delta in delta_t:
        #Make sure that delta between values are greater than zero
        if int(delta) == 0:
            delta_t_tuned.append(1)
        else:
            delta_t_tuned.append(int(delta))

    #Get index values and make sure that the last index is less than n
    long_index = np.cumsum(delta_t_tuned)
    index = [value for value in long_index if value < n]
    
    if return_index:
        return index
    else:
        pass
    
    new_time = series.time[index]
    new_value = series.value[index]

    # re-index the time series
    indexed_series = series.copy()
    indexed_series.time = new_time
    indexed_series.value = new_value

    return indexed_series
