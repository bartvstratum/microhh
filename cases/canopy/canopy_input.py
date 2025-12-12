import numpy as np
import netCDF4 as nc

# Available in microhh/python
import microhh_tools as mht

float_type = np.float32


ini = mht.Read_namelist('canopy.ini')

# Initial vertical profiles.
zsize = ini['grid']['zsize']
ktot = ini['grid']['ktot']

dz = zsize / ktot
z = np.arange(dz/2, zsize, dz)

u  = np.zeros(ktot) + 6
v  = np.zeros(ktot)


# Plant area density.
hc = 10.
pai = 0.479     # Integrated plant area density. 

pad = np.zeros(ktot)
padh = np.zeros(ktot)



# Create case_input.nc
nc_file = nc.Dataset('canopy_input.nc', mode='w', datamodel='NETCDF4', clobber=True)

nc_file.createDimension('z', kmax)
nc_z  = nc_file.createVariable('z' , float_type, ('z'))

nc_group_init = nc_file.createGroup('init');
nc_u  = nc_group_init.createVariable('u' , float_type, ('z'))
nc_v  = nc_group_init.createVariable('v' , float_type, ('z'))

nc_z [:] = z [:]
nc_u [:] = u [:]
nc_v [:] = v [:]

nc_file.close()
