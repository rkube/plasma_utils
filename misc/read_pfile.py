#-*- Coding: UTF-8 -*-

import numpy as np

def read_pfile(fname=None):
    """Read an experimental p-file.
    These files have the following format:
    NN PSI f(PSI) f'(PSI)

    with:
    NN : Number of data points for the following profile
    PSI : Radial coordinate 
    f(PSI) : profile 
    f'(PSI) : derivative of profile

    The profiles repeat

    """

    assert(fname is not None)

    data = {}

    with open(fname, "r") as df:
        var_list = []
        line = True
        while(line):
            # Read first line
            line = df.readline()
            # If we parsed the file correctly, the line below catches the first bad read
            # after processing all lines
            if (not line):
                break

            line_split = line.split()
            # First item in the first line is a number. 
            # The other column tabs are variable names
            try:
                num_rows = int(line_split[0])
                key_list = line_split[1:]

            except:
                print("Could not parse header line: {0:s}".format(line))

            if(len(key_list) > 3):
                key_list = key_list[:3]
    
            num_vars = len(key_list)
            #print("{0:d} rows, {1:d} vars: ".format(num_rows, num_vars), key_list)

            # Read num_rows rows
            for key in key_list:
                data[key] = np.zeros(num_rows)

            for row in range(num_rows):
                line = df.readline()
                numbers = [float(i) for i in line.split()]
                for kidx, key in enumerate(key_list):
                    data[key][row] = numbers[kidx]
             

    return data
    



# End of file read_pfile.py
