import numpy as np
import re, sys


def create_model(size=32, shape='square'):
    size = int(size)
    model = np.ones([size, size, 1])

    if re.match('square', shape, re.IGNORECASE) is not None:
        print("Create [square] model\n")

    elif re.match('circle', shape, re.IGNORECASE) is not None:
        cc = (size/2 -0.5, size/2 -0.5)
        mx, my = np.meshgrid(np.arange(size), np.arange(size))
        rr = (size//2)**2
        model[ (mx - cc[0])**2 + (my - cc[1])**2 > rr ] = 0
        model[ (mx - cc[0])**2 + (my - cc[1])**2 > rr ] = 0
        print("Create [circle] model\n")

    elif re.match('triangle', shape, re.IGNORECASE) is not None:
        mx, my = np.meshgrid(np.arange(size), np.arange(size))
        model[ mx >  0.5814 * my + 0.5 * size -0.5 ] = 0
        model[ mx < -0.5814 * my + 0.5 * size -0.5 ] = 0
        model[ my > 0.86 * size ] = 0
        print("Create [triangle] model\n")

    else:
        print('Unknown model shape "{}"! Please use one of the folowing:'.format(shape))
        print(' Square  |  Circle  |  Triangle\n')
        sys.exit(0)


    return model


# Custom function to parse a list of floats
def Culist(string):
    try:
        float_values = [float(value) for value in string.split(',')]
        if len(float_values) != 3:
            raise argparse.ArgumentTypeError("List must contain exactly three elements")
        return float_values
    except ValueError:
        raise argparse.ArgumentTypeError("List must contain valid floats")

