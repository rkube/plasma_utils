#-*- Coding: UTF-8 -*-

def parse_units(sim_dir):
    """Parses the units.m file which accompanies an XGC simulation

    Input:
    ======
    sim_dir: string, name of the simulation directory


    Output:
    =======
    units: dictionary with the identifiers as keys and their corresponding values
    """

    from os.path import join

    units = {}

    with open(join(sim_dir, "units.m")) as df:
        u_lines = df.readlines()
        df.close()



    for line in u_lines:
        #Split lines on whitespaces and remove any leading or trailing equal sign
        _ll = [l.strip("=") for l in line.split()]
        _ll = [l for l in _ll if (len(l) > 1) or (len(l) == 1 and l.isalnum())]

        try:
            units[_ll[0]] = float(_ll[1])
        except:
            print("Could not parse line: ", line)


    return units


def magnetic_equilibrium(sim_dir):
    """Returns psi_rz and corresponding plot ranges from the equilibrium file.

    Use case:
    >>>rr, zz = np.meshgrid(res[0], res[1])
    >>>plt.contourf(rr, zz, res[2], 32)
    >>>plt.colorbar()

    Input:
    ======
    sim_dir: string, Simulation directory

    Output:
    =======
    r_range: ndarray(float)
    z_range: ndarray(float)
    psi_rz: psi
    """

    import adios2
    from os.path import join
    import numpy as np

    df_fname = join(sim_dir, "xgc.equil.bp")
    with adios2.open(df_fname, "r") as fh:
        r_min = fh.read("eq_min_r")
        r_max = fh.read("eq_max_r")
        mr = fh.read("eq_mr")
        mz = fh.read("eq_mz")
        z_min = fh.read("eq_min_z")
        z_max = fh.read("eq_max_z")
        psi_rz = fh.read("eq_psi_rz")

        fh.close()

    r_range = np.linspace(r_min, r_max, mr)
    z_range = np.linspace(z_min, z_max, mz)

    return (r_range, z_range, psi_rz)



# End of file parse_units.py