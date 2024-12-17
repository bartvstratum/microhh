import numpy as np
import netCDF4 as nc
import xarray as xr
from numba import jit, prange
import sys
import glob
from datetime import datetime
import asyncio

# Avaiable in `microhh/python`:
import microhh_lbc_tools as mlt
import microhh_tools as mht


# Get domain name from cmd line, either "inner" or "outer".
domain = sys.argv[1]


"""
General settings.
"""
dtype = np.float64
n_ghost = 3
n_sponge = 3
lbc_freq = 30


"""
No vertical nesting for now!
"""
ktot = 64
zsize = 3200
dz = zsize / ktot


"""
Grid settings outer domain.
"""
itot_parent = 64
jtot_parent = 64

xsize_parent = 6400
ysize_parent = 6400

dx_parent = xsize_parent / itot_parent
dy_parent = ysize_parent / jtot_parent


"""
Grid settings inner domain.
"""
refinement_fac = 3

xstart_child = 1600
ystart_child = 1600

xsize_child = 3200
ysize_child = 3200

xend_child = xstart_child + xsize_child
yend_child = ystart_child + ysize_child

dx_child = dx_parent / refinement_fac
dy_child = dy_parent / refinement_fac

itot_child = int(xsize_child / dx_child)
jtot_child = int(ysize_child / dy_child)


"""
Define initial fields/profiles.
"""
dthetadz = 0.003

z  = np.arange(0.5*dz, zsize, dz)
zh = np.arange(0, zsize, dz)

u  = np.zeros(np.size(z)) + 1
v  = np.zeros(np.size(z)) + 1
th = 300. + dthetadz * z


"""
Write case_input.nc
"""
nc_file = nc.Dataset('drycblles_input.nc', mode='w', datamodel='NETCDF4', clobber=True)

nc_file.createDimension('z', ktot)
nc_z  = nc_file.createVariable('z' , dtype, ('z'))

nc_group_init = nc_file.createGroup('init');
nc_u  = nc_group_init.createVariable('u' , dtype, ('z'))
nc_v  = nc_group_init.createVariable('v' , dtype, ('z'))
nc_th = nc_group_init.createVariable('th', dtype, ('z'))

nc_z [:] = z [:]
nc_u [:] = u [:]
nc_v [:] = v [:]
nc_th[:] = th[:]

nc_file.close()


"""
Update .ini file.
"""
ini = mht.Read_namelist('drycblles.ini.base')

ini['grid']['ktot'] = ktot
ini['grid']['zsize'] = zsize
ini['cross']['sampletime'] = lbc_freq

if domain == 'outer':

    xz, yz = mlt.get_cross_locations_for_lbcs(
            xstart_child,
            ystart_child,
            xend_child,
            yend_child,
            dx_parent,
            dy_parent,
            dx_child,
            dy_child,
            n_ghost,
            n_sponge)

    ini['grid']['itot'] = itot_parent
    ini['grid']['jtot'] = jtot_parent

    ini['grid']['xsize'] = xsize_parent
    ini['grid']['ysize'] = ysize_parent

    ini['cross']['xz'] = list(xz)
    ini['cross']['yz'] = list(yz)

    ini['pres']['sw_openbc'] = False
    ini['boundary_lateral']['sw_openbc'] = False


elif domain == 'inner':

    ini['grid']['itot'] = itot_child
    ini['grid']['jtot'] = jtot_child

    ini['grid']['xsize'] = xsize_child
    ini['grid']['ysize'] = ysize_child

    # Vertical crosses through center of domain.
    ini['cross']['yz'] = (xend_child+xstart_child)/2
    ini['cross']['xz'] = (yend_child+ystart_child)/2

    ini['pres']['sw_openbc'] = True
    ini['boundary_lateral']['sw_openbc'] = True
    ini['boundary_lateral']['sw_sponge'] = n_sponge > 0
    ini['boundary_lateral']['n_sponge'] = n_sponge
    ini['boundary_lateral']['loadfreq'] = lbc_freq


ini.save('drycblles.ini', allow_overwrite=True)


"""
Create lateral boundaries for child.
"""
if domain == 'inner':

    fields = ['u', 'v', 'w', 'th', 's']
    
    # Read cross-sections.
    xz = {}
    yz = {}
    for fld in fields:
        xz[fld] = xr.open_dataset(f'outer/{fld}.xz.nc', decode_times=False)
        yz[fld] = xr.open_dataset(f'outer/{fld}.yz.nc', decode_times=False)
    time = xz[list(xz.keys())[0]].time.values
    
    # Get xarray Dataset with coordinates of LBCs.
    lbc_ds = mlt.get_lbc_xr_dataset(
            fields,
            xsize_child,
            ysize_child,
            itot_child,
            jtot_child,
            z,
            zh,
            time,
            n_ghost,
            n_sponge,
            xstart_child,
            ystart_child,
            dtype)
    
    for fld in fields:
        for loc in ['north', 'west', 'east', 'south']:
            mlt.interp_lbcs_with_xr(
                lbc_ds,
                fld,
                loc,
                xz,
                yz,
                'nearest',
                dtype)

    # Divergence check. Quick-and-dirty: only if dz is constant, dx == dy, rho=1, and domain is square :-).
    #n = n_ghost
    #time = lbc_ds['time'].values
    #for t in range(time.size):

    #    u_west = lbc_ds['u_west'][t, :, n:-n,  n].values
    #    u_east = lbc_ds['u_east'][t, :, n:-n, -n].values
    #
    #    v_south = lbc_ds['v_south'][t, :,  n, n:-n].values
    #    v_north = lbc_ds['v_north'][t, :, -n, n:-n].values
    #
    #    div_u = (u_east  - u_west).sum()
    #    div_v = (v_north - v_south).sum()
    #    div = div_u + div_v
    #
    #    print(f't={time[t]}: div_u={div_u}, div_v={div_v}, sum={div}')

    # Write LBCs as binary input for MicroHH.
    mlt.write_lbcs_as_binaries_old(lbc_ds, dtype, output_dir='.')
