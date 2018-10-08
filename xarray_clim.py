#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 11:02:39 2018

This module contains wrapper functions for xarray. The functions are 
designed to make typical in analyzing gridded climate and weather data
as easy as possible. The functions are desinged to closely resemble
the widely used command line tool CDO (Climate Data Operators)

All functions are desinged such that they work equally well with lazily
loaded dask-arrays in the background (see http://xarray.pydata.org/en/stable/dask.html).

This means that when a lazily loaded array is used as input, the output is also
lazy and wont be evaluated yet. This allows to use input datasets that are larger
then the available memory.


All functions exclusively use xarrays named indexing, thus they do not
rely on the order of the dimensions (e.g. (time,lat,lon) or (time,lon,lat)).
It is also ok if there are more dimensions present, they will simply be left
unlatered and carried through.
The testing has however only been done for the most common ordering (time,lev,lat,lon)

Most of the functions work both on xarray.Dataset and xarray.DataArray

Pros combared to cdo:
        cdo is limited to 4-dimensions, this code works for N dimensions
        directly in python



Does NOT yet work with rotated grids (lat and lon are expected to be 1d vectors)

requirements:
    xarray >= 0.10.8 (lower does not work)
    pandas (developed with 0.20.3, might also work with lower)
    numpy  (developed with 1.12.1, might also work with lower)
    dask





wirtten in python3

@author: Sebastian Scher (seb1000 on github)

Feedback is highly appreciated and can be sent to sebastian.scher@misu.su.se

This work is distributed under the GNU GPLv3 license.
"""

import calendar
import warnings

import xarray as xr
import dask
import numpy as np
import pandas as pd





def standardize_dataset(ds):
    ''' change dimension names to standard names (lat,lon,lev,time),
    ds: DataSet or DataArray
    return: dataset or datarray with standardized names
    '''
    pairs = {'latitude':'lat',
             'longitude':'lon',
             'level':'lev'}
    for key in pairs.keys():
        if key in ds.dims.keys():
            ds = ds.rename({key:pairs[key]})

    # extract variable name
    var_keys = ds.data_vars.keys()
    assert ( len(var_keys) ==1)
    for e in var_keys:
        name = e
    ds.attrs['varname'] = name


    return ds



def _lat_is_increasing(data):
    
    if data.lat[1] > data.lat[0]:
        return True
    else:
        return False

def sellonlatpoint(data,lon,lat, method='nearest'):
    '''select a single spatial point, either by taking the closest
    available point (method='nearest'), or by 
        interpolating ('linear' or 'cubic')
    '''
    
    supported_methods = ('nearest','linear','cubic')
    if method not in supported_methods:
        raise ValueError('"method" must be one of ' + str(supported_methods))
        
    # check wehther the requested point is in the domain, if not raise an error
    lonrange = (float(data.lon.min()), float(data.lon.max()))
    if lon < lonrange[0] or lon > lonrange[1]:
        raise ValueError('''the requested lon {} is outside the range of the 
                         dataset {}'''.format(lon, lonrange))
    
    latrange = (float(data.lat.min()), float(data.lat.max()))
    if lat < latrange[0] or lat > latrange[1]:
        raise ValueError('''the requested lat {} is outside the range of the 
                         dataset {}'''.format(lat, latrange))  
        
        
    if method == 'nearest':
        sub = data.sel(lat=lat,lon=lon,method=method)
    elif method in ('linear', 'cubic'):
        sub = data.interp(lat=lat,lon=lon,method=method)
    return sub

def sellonlatbox(data,west,north,east,south, allow_single_point=False):
    '''
    select a rectangular box.
    if west==east and north==south, then a single point will be returned
    if allow_single_point=True, otherwise an error will be raised
    '''
    if north == south and west==east:
        if allow_single_point:
            # in this case we return a single point
            return sellonlatpoint(data,west,north)
        else:
            raise ValueError('''the requsted area is a single point 
                             (north==south and east==west)''')
    
    if north < south:
        raise ValueError('north ({}) is smaller than south ({})'.
                         format(north,south))
        
    if north < south:
        raise ValueError('west ({}) is larger than east ({})'.
                         format(west,east))
        
    # check wehther the requested point is in the domain
    lonrange = (float(data.lon.min()), float(data.lon.max()))
    if ( west < lonrange[0] or west > lonrange[1] or
         east < lonrange[0] or east > lonrange[1] ):
        raise ValueError('''the requested lon extend {} to {} is outside 
                         the range of the
                         dataset {}'''.format(west,east,lonrange))
    
    latrange = (float(data.lat.min()), float(data.lat.max()))
    if ( north < latrange[0] or north > latrange[1] or
         south < latrange[0] or south > latrange[1] ):
        raise ValueError('''the requested lat extend {} to {} is outside 
                         the range of the 
                         dataset {}'''.format(north,south,latrange))  
       
    # depending on whether lat is increasing or decreasing (which is 
    # different for different datasets) the indexing has to be done in 
    # a different way
    if _lat_is_increasing(data):
        indexers = {'lon':slice(west,east), 'lat':slice(south,north)}
    else:
        indexers = {'lon':slice(west,east), 'lat':slice(north,south)}

    sub = data.sel(**indexers)

    return sub

def weighted_areamean(ds):
    '''area mean weighted by cos of latitude'''
    weights = xr.ufuncs.cos(np.deg2rad(ds.lat))
    # this is 1d array. Xarray broadcasts it automatically.
    # however, we compute the normalization term explicitely.
    # the normalization term is the sum of the weights over all gridpoints.
    # because the weight is not lon dependent, is is simply the sum along one 
    # meridian times the number of lon points
    norm = np.sum(weights) * len(ds.lon)
    
    # this means we have only lat and lon
    amean = (ds*weights).sum(('lat','lon')) / norm

    return amean



def wrap360_to180(ds):
    """
    wrap longitude coordinates from 0..360 to -180..180

        
    Known Issues: with lazily loaded arrays this some leads to problems
    (the dimensions in the onderlying numpy array then suddenly have a different
    order than the dask array)
    
    """
    if type(ds) is xr.DataArray:
        orig_is_array = True
        # create to dataset with a dummy variable name
        ds = ds.to_dataset(name='__dummy__')
    else:
        orig_is_array = False
        
    # check whether the data is lazily loaded
    for _, var in ds.variables.items():
        if type(var._data) is dask.array.core.Array:
            warnings.warn('''the dataset (or part of it) is lazily loaded, this 
                          sometimes casues problems in wrap369_to180''')

    
    # wrap 0..359 to  -179...180
    new = np.array([e if e < 180 else e-360 for e in ds['lon'].values])

    ds = ds.assign(**{ 'lon' : new})
    ds = ds.reindex({'lon': np.sort(new)})

    if orig_is_array:
        # convert back to 
        ds = ds['__dummy__']

    return ds    


def corr_over_time(x,y):
    '''
        point-by point correlation of two spatial arrays x and y
        (the correlation of x[all_times,lati,loni] and y[all_times,lati,loni])
        Can be used to make correlation maps of two quantities
        
    '''
    if x.shape != y.shape:
        raise ValueError('x and y must have exactly the same dimension!')    
  
    # the numpy corrcoref function works only for one point at a time, 
    # and looping over the field is very slow. therefore we do the computation
    # on a lower level (only wieh mean and std functions)
    mx = x.mean('time')
    my = y.mean('time')
    xm, ym = x-mx, y-my
    r_num = (xm*ym).mean('time')
    r_den = xm.std('time') * ym.std('time')
    
    r = r_num / r_den
    return r


def seldate(data,date):
    '''select a singel timestep. date can be string or anything that pandas recognizes
    (e.g. Timestamps)'''
    return data.sel(time=date)

def seldaterange(data,start,stop):
    '''select a range from start to (including) stop.
    dates can be string or anything that pandas recognizes
    (e.g. Timestamps)'''
    return data.sel(time=slice(start,stop))

def selmonth(data,month):
    '''
        select all dates with the given month number (0..12) or name 
        ('jan','feb',...)
    '''
    if type(month) is int:
        if month not in list(range(1,12+1)):
            raise ValueError('month musb be in 1..12, given:{}'.format(month))
            
    elif type(month) is str:
        # convert month name to number
        try:
            month = list(calendar.month_abbr).index('feb')
        except ValueError():
            raise ValueError('''{} could not be converted to a 
                             month number'''.format(month))
            
            
    return data.isel(time=data.time.dt.month==month)
    
    

def selseas(data,seas):
    ''' select season by abbreviation (djf, mam, jja, son)'''
    # convert to uppercase (xarray needs uppercase)
    seas = seas.upper()
    return data.isel(time=data.time.dt.season==seas)


def resample_time(data,freq,func='mean', **kwargs):
    '''
        resample a dataarray or dataset to a new timefrequency
        data: input dataset/dataarray
        freq: new timem frequency
        func: how to aggregate (e.g. 'mean','max')
        **kwargs: arguments passed on to xr.DataArray.resample
    
    '''
    
    if freq  in('season','seas'):
        #  due to a restriction in pandas, 'seas' is not available in resample
        # however, 'QS-DEC' gets us the meteorological seasons
        freq = 'QS-DEC'
    
    else:
        # convert from typical names like 'monht' to pandas syntax
        convert_dict = {'month':'m',
                        'day':'d',
                        'year':'yr'}
        if freq in convert_dict.keys():
            freq = convert_dict[freq]
        
    
        
    # we need to do tow steps: resample, and then apply the aggregation function   
    # defined by 'func'     
    resampled = data.resample(time=freq, **kwargs)
    try:
        # apply function for resamplein (mean, max ,...etc)
        resampled_collapsed = getattr(resampled,func)(dim='time')
    except AttributeError:
        raise ValueError('func {} not available'.format(func))
        
    assert(len(resampled_collapsed.shape)==len(data.shape))    
        
    return resampled_collapsed
    

def flexible_clim(data,freq,func='mean'):
    
    resampled = resample_time(data,freq,func)
    
    return resampled.groupby('time.'+str(freq)).mean(dim='time')

def ymon_func(data,func='mean'):
    
    return flexible_clim(data,'month',func)


def yseas_func(data,func='mean'):
    
    return flexible_clim(data,'seas',func)


#%% now some convenience wrappers for the yseas functions 
def ymonmean(data):
    return ymon_func(data,'mean')

def ymonmax(data):
    return ymon_func(data,'max')

def ymonmin(data):
    return ymon_func(data,'min')


def yseasmean(data):
    '''seasonal climatology'''
    return yseas_func(data,'mean')

def yseasmax(data):
    '''seasonal climatology'''
    return yseas_func(data,'max')

def yseasmin(data):
    '''seasonal climatology'''
    return yseas_func(data,'min')



def sellev(data,lev):
    '''select level by value'''
    return data.sel(lev=lev)

def selilev(data,lev):
    '''select level by index'''
    return data.isel(lev=lev)


def area_func(data,func='mean', **kwargs):
    '''
        **kwargs: passed on to func (e.g. q for quantile)
    '''
    try:
        # apply function 'func' which should be a funtion of xr.DataArray
        return getattr(data,func)(dim=('lat','lon'), **kwargs)
    except AttributeError:
        raise ValueError('func {} not available'.format(func))
        
        
def areamean(data):
    return area_func(data,'mean')

def areamax(data):
    return area_func(data,'max')

def areamin(data):
    return area_func(data,'min')

def areaquantile(data,q):
    return area_func(data,'quantile', q=q)

def areapercentile(data,p):
    return area_func(data,'quantile', q=p/100.)
        