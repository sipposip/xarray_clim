# xarray-clim
wrapper functions for xarray to perform common tasks in analyzing gridded climate and weather data.

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
