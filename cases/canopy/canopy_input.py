import matplotlib.pyplot as plt

import numpy as np
import netCDF4 as nc
from scipy.special import erf

# Available in microhh/python
import microhh_tools as mht

plt.close('all')

float_type = np.float32


def pad_profile(z, dz, hc=10.0, pai=0.479, sigma=0.5):
    """
    PAD profile with smooth canopy top using error function.
    Profile is scaled such that the integral of PAD matches the input PAI.
    """
    a_rel = 0.5 * (1 - erf((z - hc) / sigma))
    integral = np.sum(a_rel[z < hc + 3*sigma]) * dz
    a_base = pai / integral
    return a_rel * a_base



ini = mht.Read_namelist('canopy.ini')

# Initial vertical profiles.
zsize = ini['grid']['zsize']
ktot = ini['grid']['ktot']

dz = zsize / ktot
z = np.arange(dz/2, zsize, dz)
zh = np.arange(0, zsize+0.1, dz)

u  = np.zeros(ktot) + 6
v  = np.zeros(ktot)


# Plant area density at half levels. Code interpolates to full levels internally.
hc = 10.        # Canopy height (m)
pai = 0.479     # Integrated plant area density PAI = int(a * dz)
sigma = 1       # Width transition are canopy top (m)

padh = pad_profile(zh, dz, hc, pai, sigma) 


# Create case_input.nc
nc_file = nc.Dataset('canopy_input.nc', mode='w', datamodel='NETCDF4', clobber=True)

nc_file.createDimension('z', ktot)
nc_file.createDimension('zh', ktot+1)

nc_z = nc_file.createVariable('z', float_type, ('z'))
nc_zh = nc_file.createVariable('zh' , float_type, ('zh'))

nc_group_init = nc_file.createGroup('init');
nc_u = nc_group_init.createVariable('u', float_type, ('z'))
nc_v = nc_group_init.createVariable('v', float_type, ('z'))
nc_ph = nc_group_init.createVariable('padh', float_type, ('zh'))

nc_z[:] = z[:]
nc_zh[:] = zh[:]

nc_u[:] = u[:]
nc_v[:] = v[:]
nc_ph[:] = padh[:]

nc_file.close()
