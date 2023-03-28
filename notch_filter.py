from math import sin, cos, exp
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np

def IdealNotchFilter(shifted_mat, points, d0):
    p = len(points)
    for x in range(-d0, d0+1):
        for y in range(-d0, d0+1):
            for d in range(p):
                v0, u0 = points[d][0], points[d][1]
                shifted_mat[u0+x][v0+y] *= 0.0
    inv_shifted = np.fft.ifftshift(shifted_mat)
    inv_mat = np.fft.ifft2(inv_shifted)
    return inv_mat

def GaussianNotchFilter(shifted_mat, points, d0):
    m, n, p = shifted_mat.shape[0], shifted_mat.shape[1], len(points)
    for u in range(m):
        for v in range(n):
            for d in range(p):
                u0, v0 = points[d][0], points[d][1]
                d1 = pow(pow(u - u0, 2) + pow(v - v0, 2), 0.5)
                d2 = pow(pow(u + u0, 2) + pow(v + v0, 2), 0.5)
                shifted_mat[u][v] *= (1 - exp(-0.5 * (d1 * d2 / pow(d0, 2))))
    inv_shifted = np.fft.ifftshift(shifted_mat)
    inv_mat = np.fft.ifft2(inv_shifted)
    return inv_mat

NoisyBall = Image.open('NoisyBall.png')

# converting to np array
np_mat_noisyball = np.array(NoisyBall)

dft_noisyball_mat = np.fft.fft2(np_mat_noisyball)

shifted_dft_noisyball_mat1 = np.fft.fftshift(dft_noisyball_mat)

# display original image and its shifted_dft
plt.figure( figsize= (10, 10))

plt.subplot(2, 2, 1)
plt.title("Original Noisyball")
plt.imshow( np_mat_noisyball , cmap = 'gray' )

plt.subplot(2, 2, 2)
plt.title("Shifted DFT Noisyball")
plt.imshow( np.log(1+np.abs(shifted_dft_noisyball_mat1)) , cmap = 'gray' )

points = [
    [78, 135.3], [78, 170.6], [177.8, 149.1], [177.8, 183.7],
]

transformed_np_mat3 = GaussianNotchFilter(shifted_dft_noisyball_mat1, points, 60.0)

diffpoints = [
    [69.7, 135.3], [69.7, 170.6], [183.3, 149.1], [180.2, 183.7]
]
transformed_np_mat4 = GaussianNotchFilter(shifted_dft_noisyball_mat1, diffpoints, 15.0)

plt.subplot(2, 2, 3)
plt.title("Filtered Shifted DFT Noisyball")
plt.imshow( np.log(1+np.abs(shifted_dft_noisyball_mat1)) , cmap = 'gray' )

plt.subplot(2, 2, 4)
plt.title("Filtered Noisyball")
plt.imshow( np.abs(transformed_np_mat4) , cmap = 'gray' )

plt.show()