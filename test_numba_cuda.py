import numba as nb
from numba import cuda
import numpy as np
import cmath as cm
import matplotlib.pyplot as plt
import time as t

from Antenna_Phased_Array import Array_Config, Array_2D



#cuda.detect()

"""@cuda.jit
def add_scalars(a, b, c):
    c[0] = a + b

dev_c = cuda.device_array( (1,), np.float32)

add_scalars[1, 1](2.0, 7.0, dev_c)

c = dev_c.copy_to_host()
print(f"2.0 + 7.0 = {c[0]}")
"""


# i = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
@cuda.jit 
def pattern(theta_p: np.array, phi_p: np.array, x_e: np.array, y_e: np.array, w_e: np.array, k: float, out):
    edx, pdx = cuda.grid(2) #edx should correspond to the current block index which scales with current element
                            #pdx should correspond to the current thread index which scales with current theta/phi pair
    gdx = cuda.grid(1)

    edx = cuda.blockIdx.x
    pdx = cuda.threadIdx.x

    #print(edx, pdx, adx)
    theta = complex(theta_p[pdx])
    phi = complex(phi_p[pdx])
    x = complex(x_e[edx])
    y = complex(y_e[edx])
    w = complex(w_e[edx])
    k_c = complex(k[0])

    arg = k_c * cm.sin(theta) * (x * cm.cos(phi) + y * cm.sin(phi) )
    mult = w * 1.0j * k_c * cm.cos(theta) * ( x * cm.cos(phi) + y * cm.sin(phi) )
    # For len(theta_p) * len(phi_p) > 1024 each thread in block will evaluate angle (theta, phi) n such that (pdx % 1024 == n) 
    
    out[gdx] = mult * cm.exp(1.0j * arg)

theta_l = np.linspace(-2*np.pi, 2*np.pi, 32)
phi_l = np.linspace(-2*np.pi, 2*np.pi, 32)
theta, phi = np.meshgrid(theta_l, phi_l)

""" x = np.arange(1, 5, 1)
y = np.arange(1, , 1)
w = np.ones((5)) """

elements = Array_Config(10, 10, 0.5, 0.5, 1).rectArray()
x = elements['x']
y = elements['y']
w = elements['w']

k = 2 * np.pi

dev_theta = cuda.to_device(theta.flatten()) # Should flatten these and reshape before summation
dev_phi = cuda.to_device(phi.flatten())
dev_x = cuda.to_device(x)
dev_y = cuda.to_device(y)
dev_w = cuda.to_device(w)
dev_k = cuda.to_device(np.array([k]))

print(len(x), theta.shape[0], theta.shape[1])
#out_shape = (len(x), theta.shape[0], theta.shape[1])
out_shape = (len(x)*theta.shape[0]*theta.shape[1])

start = t.perf_counter()
dev_out = cuda.device_array(out_shape, dtype=complex) # output from each block should be the same shape as theta
                                                      # output from each thread should be singleton
                                                      # final output should be shape: (n_elements, n_theta, n_phi)

# Final product should intelligently rescale these to be powers of 2 and not exceed device limits
num_blocks = len(x)
num_threads = theta.shape[0] * theta.shape[1]
#assert num_threads <= 1024
#print(f"Num Threads: {num_threads}")

pattern[num_blocks, num_threads](dev_theta, dev_phi, dev_x, dev_y, dev_w, dev_k, dev_out)

cuda.synchronize()
out = dev_out.copy_to_host()
#print(out.shape)

dF = np.sum(out.reshape(len(x), theta.shape[0], theta.shape[1]), axis=0)
end = t.perf_counter()
print(f"GPU Time: {end-start}s")
#print(dF.shape)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.pcolormesh(theta_l, phi_l, np.abs(dF) )

start = t.perf_counter()
radar_array = Array_2D(elements, lamb = 1)
patt = radar_array.dF_dTheta(theta, phi)
end = t.perf_counter()
print(f"CPU(1) Time: {end-start}s")


fig = plt.figure()
ax = fig.add_subplot(111)
ax.pcolormesh(theta_l, phi_l, np.abs(patt) )
plt.show()
