#!/usr/bin/env python

import re
import numpy as np
import matplotlib.pyplot as plt

""

class Geqdsk:
"""
@brief G-Eqdsk reader class
@version $Id$
Copyright &copy; 2006-2008, Tech-X Corporation, Boulder, CO
See LICENSE file for conditions of use.
The official document describing g-eqdsk files:
http://fusion.gat.com/conferences/snowmass/working/mfe/physics/p3/equilibria/g_eqdsk_s.pdf
"""
	def __init__(self, filename):
        """open geqdsk file and parse its content
        case   : Identification character string
        nw     : Number of grid points in horizontal direction (R)
        nh     : Number of vgrid points in vertical direction (Z)
        rdim   : Horizontal dimension of computational box, meters
        zdim   : Vertical dimension of computational box, meters
        rcenter: R in meter of vacuum toroidal magnetic field BCENTR
        rleft  : Minimum R in meter of rectangular computational box
        zmid   : Z of center of computational box in meter
        rmaxis : R coordinate of magnetic axis in meter
        zmaxis : Z coordinate of magnetic axis in meter
        simag  : poloidal flux at magnetic axis in Weber / rad
        sibry  : poloidal flux at the plasma boundary in Weber / rad
        bcenter: Vacuum toroidal magnetic field in Tesla at RCENTR
        current: Plasma current in Ampere
        fpol   : Poloidal current function in m-T, F = RBT on flux grid
        pres   : Plasma pressure in nt / m 2 on uniform flux grid
        ffprime: FF'(psi) in (mT)^2/(Weber/rad) on uniform flux grid
        pprime : P'(psi) in (nt/m2)/(Weber/rad) on uniform flux grid
        psirz  : Poloidal flux in Weber / rad on the rectangular grid points
        qpsi   : q values on uniform flux grid from axis to boundary
        nbbbs  : Number of boundary points
        limitr : Number of limiter points
        rbbbs  : R of boundary points in meter
        zbbbs  : Z of boundary points in meter
        rlim   : R of surrounding limiter contour in meter
        zlim   : Z of surrounding limiter contour in meter
        """

        self.data = {}
        with open(filename, "r") as df:
            lines = df.readlines()
        #2lines =  open(filename, 'r').readlines()
        # first line
        m = re.search(r'^\s*(.*)\s+\d+\s+(\d+)\s+(\d+)\s*$', lines[0])
        self.data['case'] = m.group(1)
        self.data['nw'] = int(m.group(2))
        self.data['nh'] = int(m.group(3))
        fltsPat = r'^\s*([ \-]\d\.\d+[Ee][\+\-]\d\d)([ \-]\d\.\d+[Ee][\+\-]\d\d)([ \-]\d\.\d+[Ee][\+\-]\d\d)([ \-]\d\.\d+[Ee][\+\-]\d\d)([ \-]\d\.\d+[Ee][\+\-]\d\d)\s*$'

        # 2nd line
        m = re.search(fltsPat, lines[1])
        self.data['rdim'] = float(m.group(1))
        self.data['zdim'] = float(m.group(2))
        self.data['rcenter'] = float(m.group(3))
        self.data['rleft'] = float(m.group(4))
        self.data['zmid'] = float(m.group(5))
        # 3rd line
        m = re.search(fltsPat, lines[2])
        self.data['rmaxis'] = float(m.group(1))
        self.data['zmaxis'] = float(m.group(2))
        self.data['simag'] = float(m.group(3))
        self.data['sibry'] = float(m.group(4))
        self.data['bcenter'] = float(m.group(5))

        # 4th line
        m = re.search(fltsPat, lines[3])
        self.data['current'] = float(m.group(1))
        #self.data['simag'] = float(m.group(2)), ""
        #self.data['rmaxis'] = float(m.group(4)), ""

        # 5th line
        m = re.search(fltsPat, lines[4])
        #self.data['zmaxis'] = float(m.group(1)), ""
        #self.data['sibry'] = float(m.group(3)), ""
        # read remaining data
        _data = []
        counter = 5

        while 1:
            line = lines[counter]
            m = re.match(r'^\s*[ \-]\d\.\d+[Ee][\+\-]\d\d', line)
            if not m: 
                break
            _data += eval('[' + re.sub(r'(\d)([ \-]\d\.)', '\\1,\\2', line) + ']')
            counter += 1

        nw = self.data['nw']
        nh = self.data['nh']
        self.data['fpol'] = np.array(_data[0:nw])
        self.data['pres'] = np.array(_data[nw:2 * nw])
        self.data['ffprime'] = np.array(_data[2 * nw:3 * nw])
        self.data['pprime'] = np.array(_data[3 * nw:4 * nw])
        self.data['psirz'] = np.reshape( _data[4 * nw:4 * nw + nw * nh], (nh, nw))
        self.data['qpsi']  = np.array(_data[4 * nw + nw * nh:5 * nw + nw * nh])

        line = lines[counter]
        m = re.search(r'^\s*(\d+)\s+(\d+)', line)
        nbbbs = int(m.group(1))
        limitr = int(m.group(2))
        self.data['nbbbs'] = nbbbs
        self.data['limitr'] = limitr
        counter += 1

        data = []
        while 1:
            line = lines[counter]
            m = re.search(r'^\s*[ \-]\d\.\d+[Ee][\+\-]\d\d', line)
            counter += 1
            if not m: break
            data += eval('[' + re.sub(r'(\d)([ \-]\d\.)', '\\1,\\2', line) + ']')

        self.data['rbbbs'] = np.zeros(nbbbs, np.float64)
        self.data['zbbbs'] = np.zeros(nbbbs, np.float64)

        for i in range(nbbbs):
            self.data['rbbbs'][i] = data[2 * i]
            self.data['zbbbs'][i] = data[2 * i + 1]

        self.data['rlim'] = np.zeros(limitr - 1, np.float64)
        self.data['zlim'] = np.zeros(limitr - 1, np.float64)
        
        for i in range(limitr-1):
            self.data['rlim'][i] = data[2 * nbbbs + 2 * i]
            self.data['zlim'][i] = data[2 * nbbbs + 2 * i + 1]

        self.data['psi_n'] = (self.data['psirz'] - self.data['simag']) / (self.data['sibry'] - self.data['simag'])


    def getAll(self):
        return self.data

    
    def keys(self):
        return self.data.keys()
    

    def __getitem__(self, key):
        return self.data[key]


    def get_psin_profile(self, num_r=1000):
        """Calculate the radial psi_n profile at the outboard mid-plane.

        Input:
        ======
        num_r: integer - Number of data points for the profile

        Output:
        =======
        r_rg: ndarray, float - Location of the profile data points
        psin_prof: ndarray, float - radial psi_n profile data
        """
        from scipy.interpolate import interp2d

        assert((num_r > 0) & (num_r < 10000))

        r_rg = np.linspace(self.data["rmaxis"], self.data["rlim"].max(), num_r)
        r_psi, z_psi = self.get_plot_ranges()
        psi_n = (self.data['psirz'] - self.data['simag']) / (self.data['sibry'] - self.data['simag'])

        psi_ip = interp2d(r_psi, z_psi, psi_n, kind="cubic")
        psin_mid = psi_ip(r_rg, [self.data["zmaxis"]])

        return(r_rg, psin_mid)


    def get_Bpol_profile(self, num_r=1000):
        """Calculates the B_pol profile at the outboard mid-plane.
        We assume that at outboard mid-plane, B_pol is approximately in the z-direction
        and use Formula (3.2.2) from Wesson:
        B_z = 1/R d(psi)/d(r)

        Input:
        ======
        num_r: integer - Number of data points for the profile

        Output:
        =======
        r_rg: ndarray, float - Array of location points in radial direction
        Bpol_prof: ndarray float - Magnitude of poloidal B-field at r_rg points
        """

        from scipy.interpolate import interp2d

        assert((num_r > 0) & (num_r < 10000))

        r_rg = np.linspace(self.data["rmaxis"], self.data["rlim"].max(), num_r)
        r_psi, z_psi = self.get_plot_ranges()

        # Interpolate psi on r_rg and z_rg
        psi_ip = interp2d(r_psi, z_psi, self.data['psirz'], kind='cubic')
        # Interpolate psi at the outboard midplane. 
        # Calling the interpolator using a vector (rmid) and a scalar (g['zmaxis'])
        # results in psimid being a vector
        psimid = psi_ip(r_rg, [self.data["zmaxis"]])


        # Calculate radial derivative of psi at r_rg
        dpsi_dr = (psimid[2:] - psimid[:-2]) / (r_rg[2:] - r_rg[:-2])
        sep_idx = np.argmin(np.abs(psimid - self.data["sibry"]))
        Bpol = dpsi_dr / r_rg[1:-1]

        return(r_rg[1:-1], Bpol, sep_idx)


    
    def get_plot_ranges(self):
        """Return plot ranges for r and z"""
        r_rg = np.linspace(self.data['rleft'], self.data['rleft'] + self.data['rdim'], self.data['nw'])
        z_rg = np.linspace(self.data['zmid'] - 0.5 * self.data['zdim'],
                           self.data['zmid'] + 0.5 * self.data['zdim'], self.data['nh'])

        return (r_rg, z_rg)
    

    def get_triangularity(self, mode="upper"):
        """Calculate triangularity according to http://fusionwiki.ciemat.es/wiki/Triangularity

        Input:
        ======
        mode : str, either 'upper' or 'lower'
    
        Output:
        =======
        a : float, minor radius in m
        b : float, height of plasma column in m
        delta : float, triangularity = (Rgeo - R_{upper/lower})/a
        """
        assert(mode in ["upper", "lower"])

        R0 = self.data["rmaxis"], 
        Z0 = self.data["zmaxis"]

        Rmax = self.data["rbbbs"].max()
        Rmin = self.data["rbbbs"].min()
        Zmax = self.data['zbbbs'].max()
        Zmin = self.data["zbbbs"].min()
        Rgeo = 0.5 * (Rmax + Rmin)
        a = 0.5 * (Rmax - Rmin)
        b = 0.5 * (Zmax - Zmin)

        Rlower = self.data["rbbbs"][np.argmin(self.data["zbbbs"])]
        Rupper = self.data["rbbbs"][np.argmax(self.data["zbbbs"])]

        if (mode is "upper"):
            delta = (Rgeo - Rupper) / a
        else:
            delta = (Rgeo - Rlower) / a

        return (a, b, delta)



# ################################

def main():
    import sys
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-f", "--file", dest="filename",
                      help="g-eqdsk file", default="")
    parser.add_option("-a", "--all", dest="all",
                      help="display all variables", action="store_true",)
    parser.add_option("-v", "--vars", dest="vars", 
                      help="comma separated list of variables (use '-v \"*\"' for all)", default="*")
    parser.add_option("-p", "--plot", dest="plot",
                      help="plot all variables", action="store_true",)
    parser.add_option("-i", "--inquire", dest="inquire",
                      help="inquire list of variables", action="store_true",)
 

    options, args = parser.parse_args()
    if not options.filename:
        parser.error("MUST provide filename (type -h for list of options)")

	
    geq = Geqdsk(options.filename)

    if options.inquire:
        print(geq.keys())

    if options.all:
        print(geq.getAll())

    vs = geq.keys()
    if options.vars != '*':
        vs = options.vars.split(',')

    #for v in vs:
    #    print('{0:s}: {1:s}'.format(v, str(geq[v])))

    if options.plot:
        if options.vars == '*': 
            options.vars = geq.keys()
            print(options.vars)
        else:
            vs = options.vars.split(',')
            options.vars = vs

        x = np.linspace(geq['simag'], geq['sibry'], geq['nw'])

        #dx = (xmax - xmin)/float(nx - 1)
        #x = numpy.arange(xmin, xmin + (xmax-xmin)*(1.+1.e-6), dx)
        for v in options.vars:
            if v[0] != 'r' and v[0] != 'z':
                data = geq[v]
                if len(np.shape(data)) == 1:
                    plt.figure()
                    plt.plot(x, data)
                    plt.xlabel('psi poloidal')
                    plt.ylabel(v)
                    #plt.title(geq.getDescriptor(v))

        r_rg, z_rg = geq.get_plot_ranges()

        plt.figure()
        plt.contourf(r_rg, z_rg, geq['psirz'], 32)
        plt.plot(geq['rbbbs'], geq['zbbbs'], 'w-')
        plt.plot(geq['rlim'], geq['zlim'], 'k--')
        plt.colorbar()
        plt.axis('image')
        plt.title('poloidal flux')
        plt.xlabel('R')
        plt.ylabel('Z')
        plt.show()
	

if __name__ == '__main__':
    main()
