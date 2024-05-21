''' Tests for paleobeasts.signal_models.ebm

Naming rules:
1. class: Test{filename}{Class}{method} with appropriate camel case
2. function: test_{method}_t{test_id}

Notes on how to test:
0. Make sure [pytest](https://docs.pytest.org) has been installed: `pip install pytest`
1. execute `pytest {directory_path}` in terminal to perform all tests in all testing files inside the specified directory
2. execute `pytest {file_path}` in terminal to perform all tests in the specified file
3. execute `pytest {file_path}::{TestClass}::{test_method}` in terminal to perform a specific test class/method inside the specified file
4. after `pip install pytest-xdist`, one may execute "pytest -n 4" to test in parallel with number of workers specified by `-n`
5. for more details, see https://docs.pytest.org/en/stable/usage.html
'''

import pytest
import numpy as np
import paleobeasts as pb

from paleobeasts.signal_models import ebm

class TestSignalModelsEBMIntegrate:
    @pytest.mark.parametrize('y0', [[1],[10]])
    @pytest.mark.parametrize('t_span', [(0,10),(0,100)])
    @pytest.mark.parametrize('method, kwargs', [('euler',{'dt':1}),('RK45',None)])
    def test_integrate_t0(self,t_span,y0,method,kwargs):
        '''Test integrate method'''
        def func(x):
            return 1
        forcing = pb.core.Forcing(func)
        model = ebm.EBM(forcing=forcing)
        model.integrate(t_span=t_span,y0=y0,method=method,kwargs=kwargs)

class TestSignalModelsEBMtoPyleo:
    @pytest.mark.parametrize('method, kwargs', [('euler',{'dt':1}),('RK45',None)])
    @pytest.mark.parametrize('var_names', [
                                    'T',
                                    'albedo',
                                    'absorbed_SW', 
                                    'OLR', 
                                    'solar_incoming',
                                    ['T','albedo'],
                                    ['T','albedo','absorbed_SW','OLR','solar_incoming'],
                                ])
    def test_topyleo_t0(self,method,kwargs,var_names):
        '''Test to_pyleo method'''
        def func(x):
            return 1
        forcing = pb.core.Forcing(func)
        model = ebm.EBM(forcing=forcing)
        model.integrate(t_span=(0,10),y0=[100],method=method,kwargs=kwargs)
        model.to_pyleo(var_names=var_names)