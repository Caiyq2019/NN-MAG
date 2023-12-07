import numpy as np
import random
import torch_tensorrt
import torch
import argparse

def Culist(string):
    try:
        float_values = [float(value) for value in string.split(',')]
        if len(float_values) != 3:
            raise argparse.ArgumentTypeError("List must contain exactly three elements")
        return float_values
    except ValueError:
        raise argparse.ArgumentTypeError("List must contain valid floats")


def create_trt_model(model, inch, w, dtype_item, device):
    print('create trt-model size {}'.format(w))
    trt_model = torch_tensorrt.compile(
        model, 
        inputs=[torch_tensorrt.Input((1,inch,w,w), dtype=torch.float32)],
        enabled_precisions = {dtype_item},
        device=device
        )
    return trt_model


def spin_prepare(spin_split, film, rand_seed):
    # Prapare spin orientation selections
    spin_cases = []
    spin_cases.append([ 1.0, 0.0, 0.0])   # +x
    spin_cases.append([-1.0, 0.0, 0.0])   # -x
    spin_cases.append([ 0.0, 1.0, 0.0])   # +y
    spin_cases.append([ 0.0,-1.0, 0.0])   # -y
    spin_cases.append([ 1.0, 1.0, 0.0])   # +x+y
    spin_cases.append([ 1.0,-1.0, 0.0])   # +x-y
    spin_cases.append([-1.0, 1.0, 0.0])   # -x+y
    spin_cases.append([-1.0,-1.0, 0.0])   # -x-y

    # Initialize spin state
    spin = np.empty( tuple(film.size) + (3,) )
    np.random.seed(rand_seed)

    xsplit = film.size[0] / spin_split  # x length of each split area
    ysplit = film.size[1] / spin_split  # y length of each split area

    for nx in range(spin_split):
        for ny in range(spin_split):
            
            xlow_bound = int(nx * xsplit)
            xhigh_bound = int((nx+1) * xsplit) if nx + 1 < spin_split \
                                            else film.size[0]
            
            ylow_bound = int(ny * ysplit)
            yhigh_bound = int((ny+1) * ysplit) if ny + 1 < spin_split \
                                            else film.size[1]
            
            spin_selected = spin_cases[np.random.randint(len(spin_cases))]
            
            spin[xlow_bound:xhigh_bound, ylow_bound:yhigh_bound, :] = spin_selected
    
    return spin