import numpy   as np
import struct  as st
import netCDF4 as nc4

from microhh_tools import *
nl = Read_namelist()

# Settings -------
variables  = nl.cross.crosslist
indexes    = -1         # -1 automatically determines the indexes
nx         = nl.grid.itot
ny         = nl.grid.jtot
nz         = nl.grid.ktot
starttime  = 0
endtime    = nl.time.endtime
sampletime = nl.cross.sampletime
iotimeprec = 0
nxsave     = nx
nysave     = ny
endian     = 'little'
savetype   = 'float'
# End settings ---

# Set the correct string for the endianness
if (endian == 'little'):
    en = '<'
elif (endian == 'big'):
    en = '>'
else:
    raise RuntimeError("Endianness has to be little or big")

# Set the correct string for the savetype
if (savetype == 'double'):
    sa = 'f8'
elif (savetype == 'float'):
    sa = 'f4'
else:
    raise RuntimeError("The savetype has to be float or double")

# calculate the number of iterations
niter = int((endtime-starttime) / sampletime + 1)

# load the dimensions
n   = nx*ny*nz
fin = open("grid.{:07d}".format(0),"rb")
raw = fin.read(nx*8)
x   = np.array(st.unpack('{0}{1}d'.format(en, nx), raw))
raw = fin.read(nx*8)
xh  = np.array(st.unpack('{0}{1}d'.format(en, nx), raw))
raw = fin.read(ny*8)
y   = np.array(st.unpack('{0}{1}d'.format(en, ny), raw))
raw = fin.read(ny*8)
yh  = np.array(st.unpack('{0}{1}d'.format(en, ny), raw))
raw = fin.read(nz*8)
z   = np.array(st.unpack('{0}{1}d'.format(en, nz), raw))
raw = fin.read(nz*8)
zh  = np.array(st.unpack('{0}{1}d'.format(en, nz), raw))
fin.close()

# Loop over the different variables
for crossname in variables:
    if (indexes == -1):
        index_list = get_cross_indices(crossname, 'xy')
    else:
        index_list = indexes

    if (len(index_list) == 0):
        raise RuntimeError("Can find any cross-section files for {}".format(crossname))

    crossfile = nc4.Dataset("{0}.xy.nc".format(crossname), "w")

    if(crossname == 'u'): loc = [1,0,0]
    elif(crossname=='v'): loc = [0,1,0]
    elif(crossname=='w'): loc = [0,0,1]
    else:                 loc = [0,0,0]

    locx = 'x' if loc[0] == 0 else 'xh'
    locy = 'y' if loc[1] == 0 else 'yh'
    locz = 'z' if loc[2] == 0 else 'zh'

    # create dimensions in netCDF file
    dim_x  = crossfile.createDimension(locx,   nxsave)
    dim_y  = crossfile.createDimension(locy,   nysave)
    dim_z  = crossfile.createDimension(locz,   np.size(index_list))
    dim_t  = crossfile.createDimension('time', None)
    
    # create dimension variables
    var_t  = crossfile.createVariable('time', sa, ('time',))
    var_x  = crossfile.createVariable(locx,   sa, (locx,  ))
    var_y  = crossfile.createVariable(locy,   sa, (locy,  ))
    var_z  = crossfile.createVariable(locz,   sa, (locz,  ))
    var_t.units = "Seconds since start of experiment"
    
    # save the data
    var_x[:]  = x[:nxsave] if locx=='x' else xh[:nxsave]
    var_y[:]  = y[:nysave] if locy=='y' else yh[:nysave]
    
    var_s = crossfile.createVariable(crossname, sa, ('time', locz, locy, locx,))
    
    stop = False 
    for t in range(niter):
        if (stop):
            break
        for k in range(np.size(index_list)):
            index = index_list[k]
            otime = int((starttime + t*sampletime) / 10**iotimeprec)
    
            try:
                fin = open("{0:}.xy.{1:05d}.{2:07d}".format(crossname, index, otime), "rb")
            except:
                crossfile.sync()
                stop = True
                break
    
            print("Processing %8s, time=%7i, index=%4i"%(crossname, otime, index))

            var_t[t] = otime * 10**iotimeprec
            var_z[k] = z[index] if locz=='z' else zh[index] 
    
            fin = open("{0:}.xy.{1:05d}.{2:07d}".format(crossname, index, otime), "rb")
            raw = fin.read(nx*ny*8)
            tmp = np.array(st.unpack('{0}{1}d'.format(en, nx*ny), raw))
            del(raw)
            s = tmp.reshape((ny, nx))
            del(tmp)
            var_s[t,k,:,:] = s[:nysave,:nxsave]
            del(s)
    
            fin.close()
    crossfile.close() 
