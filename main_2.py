from math import sin, cos, exp
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np

def value(x, y):
    res = sin(0.1*x) + sin(0.2*x) + cos(0.4*x) + sin(((x*x + y*y)**0.5)*0.15) + sin(((x*x + y*y)**0.5)*0.35)
    return res

# An empty image of 512 x 512 pixels.
# extra pixel created for padding the image with zeros(preventing the effects due to periodicity of the image (as taught in lectures))
#np_mat = np.zeros((1024, 1024))
np_mat = np.zeros((512, 512))

m, n = np_mat.shape[0], np_mat.shape[1]
for x in range(512):
    for y in range(512):
        np_mat[y][x] = value(x, y)

cropped_np_mat = np_mat[0:512, 0:512]

fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(2, 3, 1)
ax1.title.set_text("Original Image")
ax1.imshow(cropped_np_mat, cmap='gray')

# dft tranform of image.
fourier_transformed_img = np.fft.fft2(np_mat)

# shifting zero frequency to the center of the image.
shifted_img = np.fft.fftshift(fourier_transformed_img)

# magnitude of DFT
magnitude = np.abs(shifted_img)

# phase of DFT
phase = np.angle(shifted_img)

ax2 = fig.add_subplot(2, 3, 2)
ax2.title.set_text("Magnitude(log) of DFT")
ax2.imshow(np.log(magnitude), cmap = 'gray' )

ax3 = fig.add_subplot(2, 3, 3)
ax3.title.set_text("Phase of DFT")
plt.imshow( phase, cmap = 'gray' )

# Simply multiplying the shifted_img by 2, doubles the magnitude and has no effect on phase.
shifted_img = 2*shifted_img

# Inverse shifting modified img
inv_shifted_img = np.fft.ifftshift(shifted_img)

# Inverse fourier tranform
inv_fourier_transformed = np.fft.ifft2(inv_shifted_img)

cropped_inv_fourier_transformed = inv_fourier_transformed[0:512, 0:512]

ax4 = fig.add_subplot(2, 3, 5)
ax4.title.set_text("Transformed Image")
ax4.imshow( np.abs(cropped_inv_fourier_transformed), cmap = 'gray' )

plt.show()