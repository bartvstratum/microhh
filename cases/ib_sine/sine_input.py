import numpy as np
import netCDF4 as nc

import microhh_tools as mht

class Grid:
    def __init__(self, kmax, nloc1, nbuf1, dz1, dz2):
        """
        Class to create a smooth stretched vertical grid.
        """
        dn         = 1./kmax
        n          = np.linspace(dn, 1.-dn, kmax)
        nloc1     *= dn
        nbuf1     *= dn
        dzdn1      = dz1/dn
        dzdn2      = dz2/dn

        dzdn       = dzdn1 + 0.5*(dzdn2-dzdn1)*(1. + np.tanh((n-nloc1)/nbuf1))
        self.dz    = dzdn*dn

        self.kmax  = kmax
        self.z     = np.zeros(self.dz.size)
        stretch    = np.zeros(self.dz.size)

        self.z[0]  = 0.5*self.dz[0]
        stretch[0] = 1.

        for k in range(1, self.kmax):
              self.z [k] = self.z[k-1] + 0.5*(self.dz[k-1]+self.dz[k])
              stretch[k] = self.dz[k]/self.dz[k-1]

        self.zh = np.insert(np.cumsum(self.dz), 0, 0)
        self.zsize = self.z[kmax-1] + 0.5*self.dz[kmax-1]
        self.dz = self.zh[1:] - self.zh[:-1]

        print(f'kmax={kmax}, zsize={self.zsize:.3f}')

    def plot(self):
        plt.figure()
        plt.plot(self.z, self.dz, '-x')
        plt.xlabel('z (m)')
        plt.ylabel('dz (m)')


if __name__ == "__main__":

    plot = False

    if plot:
        import matplotlib.pylab as plt
        plt.close('all')

    # MicroHH-mode (np.float32 (single) or np.float64 (double) precision.
    float_type = np.float32

    # Domain settings.
    itot = 256
    jtot = 64

    xsize = 0.1016
    dx = xsize / itot

    dy = dx
    ysize = dy * jtot

    # Settings cosine hills (all in units [m])
    amplitude = 0.00254
    wavelength_x = xsize
    wavelength_y = 0
    z_offset = 0.002

    # Mean height of hills.
    z_mean = z_offset + amplitude

    # Create stretched grid
    #grid = Grid(16, 40, 5, 0.0008, 0.0015)
    #grid = Grid(96, 40, 5, 0.0004, 0.0007)
    grid = Grid(128, 40, 5, 0.0002, 0.0005373)
    #grid = Grid(256, 122, 10, 0.0001, 0.000322)
    #grid = Grid(384, 180, 20, 0.00006, 0.00021831)

    # Vertical grid spacing near IB.
    # IB method works best if dz ~= dx, not for dz << dx.
    k_min = np.abs(grid.z - z_mean).argmin()
    dz_ib = grid.dz[k_min]

    print(f'dx={dx*100:.2f} cm, dy={dy*100:.2f} cm, dz_ib={dz_ib*100:.2f} cm')

    if plot:
        grid.plot()

    print('Effective zsize = {}'.format(grid.zsize-z_offset-amplitude))

    """
    Update .ini file.
    """
    ini = mht.Read_namelist('sine.ini.base')

    ini['grid']['xsize'] = xsize
    ini['grid']['ysize'] = ysize
    ini['grid']['zsize'] = grid.zsize

    ini['grid']['itot'] = itot
    ini['grid']['jtot'] = jtot
    ini['grid']['ktot'] = grid.kmax

    # Normalised locations of observations by Hudson.
    x_norm = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.645, 0.75, 0.85, 0.933])
    x_cross = x_norm * xsize 

    ini['cross']['yz'] = list(x_cross)

    ini.save('sine.ini', allow_overwrite=True)

    # Create initial profiles:
    z = grid.z
    u = 0.00137 * np.ones(z.size)

    # Write the data to a .nc file for MicroHH
    nc_file = nc.Dataset('sine_input.nc', mode='w', datamodel='NETCDF4', clobber=True)
    nc_file.createDimension('z', grid.kmax)
    nc_z  = nc_file.createVariable('z' , float_type, ('z'))

    nc_group_init = nc_file.createGroup('init');
    nc_u  = nc_group_init.createVariable('u' , float_type, ('z'))

    nc_z[:] = grid.z [:]
    nc_u[:] = u[:]

    nc_file.close()

    # Create 2D height map
    dx = ini['grid']['xsize'] / ini['grid']['itot']
    dy = ini['grid']['ysize'] / ini['grid']['jtot']

    x  = np.arange(0.5*dx, ini['grid']['xsize'], dx)
    xh = np.arange(0, ini['grid']['xsize']+1e-9, dx)

    y  = np.arange(0.5*dy, ini['grid']['ysize'], dy)
    yh = np.arange(0, ini['grid']['ysize']+1e-9, dy)

    dem = np.zeros((ini['grid']['itot'], ini['grid']['jtot']), dtype=float_type)

    for j in range(ini['grid']['jtot']):
        #dem[:,j] = z_offset + amplitude + amplitude * np.sin(2*np.pi*x/wavelength_x)
        dem[:,j] = z_offset + amplitude + amplitude * np.cos(2*np.pi*x/wavelength_x)

    if plot:
        plt.figure()
        plt.plot(x, dem[:,0])

        #for k in range(z.size):
        #    plt.plot(xh, np.ones_like(xh)*grid.zh[k], 'k-', linewidth=0.5)
        #    plt.scatter(x, np.ones_like(x)*grid.z[k], s=1, color='r')
        #for i in range(x.size):
        #    plt.plot(np.ones_like(grid.zh)*xh[i], grid.zh, 'k-', linewidth=0.5)

    dem.T.tofile('dem.0000000')
