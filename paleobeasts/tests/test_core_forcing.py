''' Tests for paleobeasts.core.forcing

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

class TestSignalModelsFromCSV:
    @pytest.mark.parametrize('dataset, value_name, time_name', [('vieira_tsi',None,None),('vieira_tsi','10','Age (yrs BP)'),
                                                                ('insolation',None,None),('insolation','insol_65N_d233','kyear')])
    def test_from_csv_t0(self,dataset,value_name,time_name):
        '''Test from_sv method'''
        _ = pb.Forcing.from_csv(dataset=dataset,value_name=value_name,time_name=time_name)
