import functools
import importlib

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, interp1d

class Forcing:
    def __init__(self, data, time=None, params=None, interpolation='cubic'):
        """
        Initialize the Forcing class.

        Parameters:
        -----------
        data : callable function or array
            representing the forcing over time.

        time : numeric or array
            time values corresponding to the data points
            if data is an array, time must be provided.

        derivative : callable function representing the derivative of the forcing, or None if `data` is an array.

        """
        self.data = data
        self.time = time
        self.params = params if params is not None else {}
        self.forcing_type = None


        if isinstance(self.data, np.ndarray):
            print('data is an array')
            self.forcing_type = 'interpolated array {}'.format(interpolation)
            if interpolation == 'cubic':
                interp_func = CubicSpline
            elif interpolation == 'linear':
                interp_func = interp1d

            if self.time is not None:
                self.forcing_func = interp_func(time, data, **self.params)
            else:
                self.forcing_func = interp_func(time, data, **self.params)
        elif callable(self.data):
            self.forcing_type = 'function'
            self.forcing_func = functools.partial(self.data, **self.params)

    @classmethod
    def from_csv(self,dataset=None,file_path=None,value_name=None,time_name=None,params=None,interpolation='cubic'):
        '''Function to create a forcing object from a csv file
        
        Parameters
        ----------
        
        dataset : str; {'vieira_tsi', 'insolation'}
            Name of the dataset. If None, then file_path must be provided.
            Currently 'vieira_tsi' and 'insolation' are supported.
                - vieira_tsi: Vieira et al. (2011) TSI reconstruction. Default value_name is '1' (first realization), default time_name is 'Age (kyrs BP)'.
                - insolation: Insolation data from climlab. Default value_name is 'insol_65N_d172', default time_name is 'kyear'.

        file_path : str
            Path to the csv file. If None, then dataset must be provided.

        time_name : str
            Name of the column containing the time data. If None, then the index will be used.
        
        value_name : str
            Name of the column containing the forcing data. If None, then the first column will be used.

        params : dict
            Parameters to pass to the interpolation function.

        interpolation : str; {'cubic', 'linear'}
            Type of interpolation to use. Default is 'cubic'.
        '''

        if dataset is not None:
            if dataset == 'vieira_tsi':
                my_resources = importlib.resources.files("paleobeasts") / "data"
                file_path = my_resources.joinpath("vieira_tsi.csv")
                df = pd.read_csv(file_path)

                #Load default time and value
                if time_name is None:
                    time_name = 'Age (kyrs BP)'
                if value_name is None:
                    value_name = '0'
            elif dataset == 'insolation':
                my_resources = importlib.resources.files("paleobeasts") / "data"
                file_path = my_resources.joinpath("insolation.csv")
                df = pd.read_csv(file_path)

                if time_name is None:
                    time_name = 'kyear'
                if value_name is None:
                    value_name = 'insol_65N_d172'
            else:
                raise ValueError('Dataset not recognized')
        else:
            df = pd.read_csv(file_path)
        
        forcing = Forcing(data=df[value_name].values,time=df[time_name].values,params=params,interpolation=interpolation)

        return forcing

    def get_forcing(self, t):
        """
        Get the forcing value at time t.
        """
        return self.forcing_func(t)
        # if callable(self.data):
        #     return self.data(t, **self.params)
        # elif isinstance(self.data, np.ndarray):
        #     idx = int(t)  # TODO Assuming t is an index; interpolate if not
        #     return self.data[idx]

