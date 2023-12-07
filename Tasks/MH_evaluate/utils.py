import cv2
import numpy as np
import random
import torch
import argparse
import re, sys


def LL(x):
    return torch.where(x >= 0, torch.log(x+1), -torch.log(-x+1))



def create_random_mask(shape, num_points, inverse=False):
    size=shape[0]
    points = np.random.randint(0, size, size=(num_points, 2))
    hull = cv2.convexHull(points)
    image = np.zeros((size, size), dtype=np.uint8)
    cv2.drawContours(image, [hull], 0, 255, -1)

    # 进行二值化处理
    _, binary_image = cv2.threshold(image, 0, 1, cv2.THRESH_BINARY)

    if inverse:
        binary_image = 1 - binary_image
    
    # 定义目标数组形状
    new_shape = shape + (1, 3)
    # 将原始数组赋值给目标数组的三个通道
    new_arr = np.zeros(new_shape, dtype=binary_image.dtype)
    new_arr[..., :] = binary_image[..., np.newaxis, np.newaxis]

    return new_arr



def create_regular_mask(shape, mask_type='square'):
    size = shape[0]
    mask = np.ones([size, size])

    if re.match('square', mask_type, re.IGNORECASE) is not None:
        pass

    elif re.match('circle', mask_type, re.IGNORECASE) is not None:
        cc = (size/2 -0.5, size/2 -0.5)
        mx, my = np.meshgrid(np.arange(size), np.arange(size))
        rr = (size//2)**2
        mask[ (mx - cc[0])**2 + (my - cc[1])**2 > rr ] = 0
        mask[ (mx - cc[0])**2 + (my - cc[1])**2 > rr ] = 0

    elif re.match('triangle', mask_type, re.IGNORECASE) is not None:
        mx, my = np.meshgrid(np.arange(size), np.arange(size))
        mask[ mx >  0.5814 * my + 0.5 * size -0.5 ] = 0
        mask[ mx < -0.5814 * my + 0.5 * size -0.5 ] = 0
        mask[ my > 0.86 * size ] = 0

    elif re.match('hole', mask_type, re.IGNORECASE) is not None:
        cc = (size/2 -0.5, size/2 -0.5)
        mx, my = np.meshgrid(np.arange(size), np.arange(size))
        rr = (size//4)**2
        mask[ (mx - cc[0])**2 + (my - cc[1])**2 < rr ] = 0
        mask[ (mx - cc[0])**2 + (my - cc[1])**2 < rr ] = 0

    else:
        print('Unknown mask type "{}"! Please use one of the folowing:'.format(mask_type))
        print(' Square  |  Circle  |  Triangle | Hole\n')
        sys.exit(0)

    # 定义目标数组形状
    new_shape = shape + (1, 3)
    # 将原始数组赋值给目标数组的三个通道
    new_arr = np.zeros(new_shape, dtype=mask.dtype)
    new_arr[..., :] = mask[..., np.newaxis, np.newaxis]

    return new_arr




def spin_prepare(spin_split, film, rand_seed, mask=False):
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


    if mask==True:
        shape=(film.size[0],film.size[1])
        inverse=random.choice([True, False])
        inverse=False
        num_points=np.random.randint(low=2, high=shape[0])
        spin_mask=create_random_mask(shape, num_points, inverse)

    elif type(mask) == str:
        shape=(film.size[0],film.size[1])
        spin_mask=create_regular_mask(shape, mask)

    else:
        spin_mask=1
    
    return spin * spin_mask
        


def winding_density(spin_batch):
    """
    用于计算batch数据的winding density
    Args:
    spin_batch: torch.tensor
                形状为(batch_size, 3, 32, 32)的tensor，表示包含batch_size个样本的spin数据
    Returns:
    winding_density_batch: torch.tensor
                形状为(batch_size, 32, 32)的tensor，表示batch数据的winding density
    """
    # 调整spin的维度顺序为[batch_size, 32, 32, 1, 3]
    spin = spin_batch.permute(0, 2, 3, 1).unsqueeze(-2)
    spin_xp = torch.roll(spin, shifts=-1, dims=1)
    spin_xm = torch.roll(spin, shifts=1, dims=1)
    spin_yp = torch.roll(spin, shifts=-1, dims=2)
    spin_ym = torch.roll(spin, shifts=1, dims=2)
    spin_xp[:, -1, :, :, :] = spin[:, -1, :, :, :]
    spin_xm[:, 0, :, :, :]  = spin[:, 0, :, :, :]
    spin_yp[:, :, -1, :, :] = spin[:, :, -1, :, :]
    spin_ym[:, :, 0, :, :]  = spin[:, :, 0, :, :]
    winding_density = (spin_xp[:,:,:, 0, 0] - spin_xm[:,:,:, 0, 0]) / 2 * (spin_yp[:,:,:, 0, 1] - spin_ym[:,:,:, 0, 1]) / 2 \
                    - (spin_xp[:,:,:, 0, 1] - spin_xm[:,:,:, 0, 1]) / 2 * (spin_yp[:,:,:, 0, 0] - spin_ym[:,:,:, 0, 0]) / 2
    
    winding_density = winding_density / np.pi
    #winding_abs = torch.abs(winding_density).sum()
    winding_abs = torch.round(
                             torch.abs(winding_density).sum(dim=(1,2))
                             )
    winding_sum = torch.round(
                             winding_density.sum(dim=(1,2))
                             )

    return winding_density.squeeze(), winding_abs.item(), winding_sum.item()



# Custom function to parse a list of floats
def Culist(string):
    try:
        float_values = [float(value) for value in string.split(',')]
        if len(float_values) != 3:
            raise argparse.ArgumentTypeError("List must contain exactly three elements")
        return float_values
    except ValueError:
        raise argparse.ArgumentTypeError("List must contain valid floats")


def MaskTp(string):
    if string == "False":
        return False
    elif string == "True":
        return True
    else:
        return string
