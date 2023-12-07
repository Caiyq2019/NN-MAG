import os, cv2
import numpy as np
import matplotlib.pyplot as plt

def initial_spin_prepare(w, layers, seed):
    np.random.seed(seed)
    matrix = np.random.randn(w, w, 2)
    matrix = matrix / np.linalg.norm(matrix, axis=2, keepdims=True)
    matrix = np.expand_dims(matrix, axis=2)
    #添加z方向的分量，值为0
    spin = np.concatenate([matrix, np.zeros((w, w, 1, 1))], axis=3)
    #直接将单层薄膜复制为多层
    spin = np.tile(spin, (1,1, layers, 1)) 
    return spin

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
    # new_shape = shape + (1, 3)
    new_shape = shape + (2, 3)
    # 将原始数组赋值给目标数组的三个通道
    new_arr = np.zeros(new_shape, dtype=binary_image.dtype)
    new_arr[..., :] = binary_image[..., np.newaxis, np.newaxis]
    return new_arr



def error_plot(data, save_path, Hext):
    # 绘制折线图
    plt.plot(data)
    plt.yscale('log')

    # 添加标题和标签
    plt.title('Spin update errors\n Hext(oe):{}'.format(Hext))
    plt.xlabel('Iterations')
    plt.ylabel('log_error')

    plt.savefig(save_path+'.png')
    plt.close()


def data_rotate(Spins, Hds, symtype=None):
    # 根据对称性，构造更多的数据
    if symtype == 'RX': # X image
        Spins_rotate = Spins[:,::-1] * np.array([-1,1,1,-1,1,1], dtype=np.float16)
        Hds_rotate = Hds[:,::-1] * np.array([-1,1,1,-1,1,1], dtype=np.float16)

    elif symtype == 'RY': # Y image
        Spins_rotate = Spins[:,:,::-1] * np.array([1,-1,1,1,-1,1], dtype=np.float16)
        Hds_rotate = Hds[:,:,::-1] * np.array([1,-1,1,1,-1,1], dtype=np.float16)

    elif symtype == 'R90': # Rotate 90
        Spins_rotate = Spins.transpose(0,2,1,3)[:,::-1].copy()
        spin_tmp = Spins_rotate.copy()
        Spins_rotate[...,0], Spins_rotate[...,1] = spin_tmp[...,1], spin_tmp[...,0] 
        Spins_rotate[...,3], Spins_rotate[...,4] = spin_tmp[...,4], spin_tmp[...,3] 
        Spins_rotate = Spins_rotate * np.array([-1,1,1,-1,1,1], dtype=np.float16)
        Hds_rotate = Hds.transpose(0,2,1,3)[:,::-1].copy()
        Hds_tmp = Hds_rotate.copy()
        Hds_rotate[...,0], Hds_rotate[...,1] = Hds_tmp[...,1], Hds_tmp[...,0] 
        Hds_rotate[...,3], Hds_rotate[...,4] = Hds_tmp[...,4], Hds_tmp[...,3] 
        Hds_rotate = Hds_rotate * np.array([-1,1,1,-1,1,1], dtype=np.float16)

    elif symtype == 'R180': # Rotate 180
        Spins_rotate = Spins[:,::-1,::-1] * np.array([-1,-1,1,-1,-1,1], dtype=np.float16)
        Hds_rotate = Hds[:,::-1,::-1] * np.array([-1,-1,1,-1,-1,1], dtype=np.float16)

    elif symtype == 'R270': # Rotate 270
        # Rotate 270
        Spins_rotate = Spins.transpose(0,2,1,3)[:,:,::-1].copy()
        spin_tmp = Spins_rotate.copy()
        Spins_rotate[...,0], Spins_rotate[...,1] = spin_tmp[...,1], spin_tmp[...,0] 
        Spins_rotate[...,3], Spins_rotate[...,4] = spin_tmp[...,4], spin_tmp[...,3] 
        Spins_rotate = Spins_rotate * np.array([1,-1,1,1,-1,1], dtype=np.float16)
        Hds_rotate = Hds.transpose(0,2,1,3)[:,:,::-1].copy()
        Hds_tmp = Hds_rotate.copy()
        Hds_rotate[...,0], Hds_rotate[...,1] = Hds_tmp[...,1], Hds_tmp[...,0] 
        Hds_rotate[...,3], Hds_rotate[...,4] = Hds_tmp[...,4], Hds_tmp[...,3] 
        Hds_rotate = Hds_rotate * np.array([1,-1,1,1,-1,1], dtype=np.float16)

    return Spins_rotate, Hds_rotate