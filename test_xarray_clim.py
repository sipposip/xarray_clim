#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 11:02:39 2018


requires: xarray >= 0.10.8

@author: sebastian
"""

import xarray as xr
import numpy as np
import pandas as pd
import os
import pytest
from xarray_clim import *

## for testing, create an artificial dataarray
dates = pd.date_range('19900101','19931231')
lats = np.arange(-90,90.1,10)
lons = np.arange(0,360,10)
ntime = len(dates)
nlats = len(lats)
nlons = len(lons)

data = np.fromfunction(lambda t,x,y: 0.001*t+np.sin(t/100.)+np.cos(x)+np.sin(y), (ntime,nlats,nlons))

data = xr.DataArray(data=data,
                    dims = ('time','lat','lon'),
                    coords={'time':dates,'lat':lats,'lon':lons})



data_decreasing_lat = xr.DataArray(data=data,
                    dims = ('time','lat','lon'),
                    coords={'time':dates,'lat':lats[::-1],'lon':lons})

os.system('rm test.nc')
data.to_dataset('var1').to_netcdf('test.nc')
data_lazy = xr.open_mfdataset('test.nc')



#%% tests
    
def test_selclosetpoint():
    
    with pytest.raises(ValueError):
        assert (sellonlatpoint(data,10,100))
        
        
    assert(sellonlatpoint(data,10,10).shape==(ntime,))
    
    
    
    
def test_corr():
    # correlation with itself must be 1 for all points
    assert(np.allclose(corr_over_time(data,data),1))
    # correlation with minus itself must be -1
    assert(np.allclose(corr_over_time(data,-data),-1))
    
    
def test_wrap():
    with pytest.warns(Warning):
        assert(wrap360_to180(data_lazy)    )