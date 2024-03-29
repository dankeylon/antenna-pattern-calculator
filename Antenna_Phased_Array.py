"""

Author: Daniel Keylon

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi
from scipy.optimize import newton, fmin
from scipy.signal import chebwin, find_peaks
from scipy.integrate import quad, dblquad
from scipy.special import binom
from scipy.interpolate import interp1d, interp2d
from typing import Literal

from numba import cuda, njit, prange
import cmath as cm


#dB Conversion
def linTodB(a):
    return 10.0 * np.log10(a)

def dBToLin(a):
    return 10.0**(a / 10.0)

#Steering Function
def a_mn(x1, y1, u_0 = 0, v_0 = 0, lamb = 1):
    return np.exp(-1j * 2 * pi / lamb * (x1 * u_0 + y1 * v_0) )

def angToUV(theta, phi):
    u = np.sin(theta) * np.cos(phi)
    v = np.cos(theta) * np.sin(phi)
    return u, v

def getSLL(pattCut):
    #Works so long as the pattern is symmetric with the mainlobe at theta=0
    #Does not work well with large grating lobes or single lobe patterns
    
    peaks = find_peaks(pattCut)[0]
    if len(peaks) >= 3:
        cntrPeakIdx = int( len(peaks)/2 )
        mainLobe = peaks[ cntrPeakIdx ]
        sideLobes = peaks[ [cntrPeakIdx - 1, cntrPeakIdx + 1] ]
    
        return pattCut[sideLobes]
    
    else:
        print('Insufficient Peaks')
        # theta = np.linspace(-pi/2, pi/2, len(pattCut) )
        # plt.plot(theta, pattCut)
        # plt.show()
        
        GLL = pattCut[-1]
        
        return [GLL]

#Normalizes voltage patterns  
def normal(array):
    return np.abs( array ) / np.max( np.abs( array ) )


##Useful Constants
c = 3*10**8
kTo = dBToLin(-204)


#Primary 2D Array Code
class Array_2D:
    #Calculates the array pattern that corresponds to an element configuraton
    #Provides methods to calculate angle pattern, u-v pattern, gain pattern, Beam Solid Angle
    
    def __init__(self, elements, lamb = 1):
        #Elements is a dictionary {"x": [],"y":[] ,"w": []}

        self.elements = elements
        self.lamb     = lamb
        self.k        = (2*pi)/lamb
        
        #Calculate Approximate Max
        theta = np.linspace(-pi/2, pi/2, 1000)
        patt = self.arrayFactor(theta, 0)
        self.F_max = np.max( np.abs(patt) )
        
        
    def arrayFactor(self, theta, phi):

        F = 0
        
        x = self.elements["x"]
        y = self.elements["y"]
        w = self.elements["w"]
        
        for idx in range(0, len(x) ):
            
            arg = self.k * np.sin(theta) * ( x[idx] * np.cos(phi) + y[idx] * np.sin(phi) )
            F += w[idx] * np.exp(1j * arg)
                        
        return F
    
    def dF_dTheta(self, theta, phi, device: Literal['cpu_single', 'cpu_numba', 'cpu_multi', 'gpu'] = "cpu_single"):
        x = self.elements["x"]
        y = self.elements["y"]
        w = self.elements["w"]

        # Constrain Memory Usage to be below system limits
        # item_size = np.dtype(dtype).itemsize
        # total_elements = np.prod(shape)
        
        # TODO: Intelligently decide based on manual selection, gpu availability, and projected t_calc (empirical)
        if device == "gpu":
            @cuda.jit 
            def pattern(theta_p, phi_p, x_e, y_e, w_e, k, out): # TODO: Type Hints?
                                        
                block_idx = cuda.blockIdx.x   # block_idx corresponds to the current block index which scales with current element
                thread_idx = cuda.threadIdx.x # thread_idx corresponds to the current thread index, thread_idx in [0, 1024]

                x = complex(x_e[block_idx])
                y = complex(y_e[block_idx])
                w = complex(w_e[block_idx])
                k_c = complex(k[0])

                num_angles = len(theta_p)

                for angle_idx in range(thread_idx, num_angles, cuda.blockDim.x):
                    theta = complex(theta_p[angle_idx])
                    phi = complex(phi_p[angle_idx])

                    arg = k_c * cm.sin(theta) * (x * cm.cos(phi) + y * cm.sin(phi) )
                    mult = w * 1.0j * k_c * cm.cos(theta) * ( x * cm.cos(phi) + y * cm.sin(phi) )
                    out[block_idx * num_angles + angle_idx] = mult * cm.exp(1.0j * arg)

            dev_theta = cuda.to_device(theta.flatten())
            dev_phi = cuda.to_device(phi.flatten())
            dev_x = cuda.to_device(x)
            dev_y = cuda.to_device(y)
            dev_w = cuda.to_device(w)
            dev_k = cuda.to_device(np.array([self.k]))
            out_shape = (len(x)*theta.shape[0]*theta.shape[1])
            dev_out = cuda.device_array(out_shape, dtype=complex) # output from each block should be linear array the same length as theta
                                                                  # output from each thread should set a single entry in out_shape
                                                                  # final output should be shape: (n_elements*n_theta*n_phi,)

            # TODO: Final product should intelligently rescale these to be powers of 2 and not exceed device limits
            num_blocks = len(x)
            num_threads = min(theta.shape[0] * theta.shape[1], 1024)
            pattern[num_blocks, num_threads](dev_theta, dev_phi, dev_x, dev_y, dev_w, dev_k, dev_out)

            cuda.synchronize()
            out = dev_out.copy_to_host()

            dF = np.sum(out.reshape(len(x), theta.shape[0], theta.shape[1]), axis=0)

        elif device == "cpu_single":
            # Perform all operations on a single CPU thread
            def pattern(x, y, w):
                arg = self.k * np.sin(theta) * (x * np.cos(phi) + y * np.sin(phi) )
                mult = w * 1j * self.k * np.cos(theta) * ( x * np.cos(phi) + y * np.sin(phi) )
                return mult * np.exp(1j * arg)
        
            dF = np.sum([pattern(xi, yi, wi) for xi, yi, wi in zip(x, y, w)], axis=0)

        elif device == "cpu_numba":
            @njit(parallel=False)
            def pattern(theta, phi, x_e, y_e, w_e, k):
                dF = np.zeros(theta.shape, dtype=np.complex128)
                mult = np.zeros(theta.shape, dtype=np.complex128)
                arg = np.zeros(theta.shape, dtype=np.complex128)
                for x, y, w in zip(x_e, y_e, w_e):
                    arg = k * np.sin(theta) * (x * np.cos(phi) + y * np.sin(phi) )
                    mult = w * 1j * k * np.cos(theta) * ( x * np.cos(phi) + y * np.sin(phi) )
                    dF += mult * np.exp(1j * arg)
                return dF
            
            dF = pattern(theta, phi, x, y, w, self.k)

        elif device == "cpu_multi":
            @njit(parallel=True)
            def pattern(theta, phi, x_e, y_e, w_e, k):
                dF = np.zeros(theta.shape, dtype=np.complex128)
                mult = np.zeros(theta.shape, dtype=np.complex128)
                arg = np.zeros(theta.shape, dtype=np.complex128)
                for edx in prange(0, len(x_e)):
                    x = x_e[edx]
                    y = y_e[edx]
                    w = w_e[edx]

                    arg = k * np.sin(theta) * (x * np.cos(phi) + y * np.sin(phi) )
                    mult = w * 1j * k * np.cos(theta) * ( x * np.cos(phi) + y * np.sin(phi) )
                    dF += mult * np.exp(1j * arg)
                return dF
            
            dF = pattern(theta, phi, x, y, w, self.k)
        
        return dF
    
    def dF_dPhi(self, theta, phi):
        
        dF = 0
        
        x = self.elements["x"]
        y = self.elements["y"]
        w = self.elements["w"]
        
        for idx in range(0, len(x) ):
            
            arg = self.k * np.sin(theta) * (x[idx] * np.cos(phi) + y[idx] * np.sin(phi) )
            mult = w[idx] * 1j * self.k * np.sin(theta) * ( y[idx] * np.cos(phi) - x[idx] * np.sin(phi) )
            
            dF += mult * np.exp(1j * arg)
            
        return dF
        
    def uniPattUV(self, u, v):

        F = 0
        
        x = self.elements["x"]
        y = self.elements["y"]
        w = self.elements["w"]
        
        for idx in range(0, len(x)):
            
            arg = self.k * (x[idx]*u + y[idx]*v) 
            F = F + w[idx] * np.exp(1j * arg)
        
        return F
    
    def gainPeak(self, F):
        
        self.G_peak = (4 * pi) / self.beamSA(F)
        
        return self.G_peak
    
    def beamSA(self, F):
        
        Fxn = lambda phi, theta: np.abs( F(theta,phi) )**2 / np.abs( self.F_max )**2 * np.sin(theta)
        
        Omega = dblquad( Fxn, 0, pi, lambda phi: 0, lambda phi: 2*pi )[0]

        return Omega
    
    def gainPatt(self, F, peak):
        
        G = lambda x, y: self.G_p * np.abs( F(x, y) )**2 / self.F_max**2
    
        return G
    
    def gainUV(self, u, v, F):
        
        du = u[1] - u[0]
        dv = v[1] - v[0]
        
        uu, vv = np.meshgrid(u, v)
        
        den = np.sqrt( (1 - uu**2 - vv**2) + 0j )
        realden = (np.imag(den) == 0) 
        
        # den = np.real(den * realden)
        
        OmegaA = np.real( np.sum( np.sum( np.abs(F)**2 / den  * realden ) ) * du * dv )
        
        G = 4 * pi / OmegaA
        
        return G
    
    def plotElementWeights(self):
        #Plots the weights of the various elements in a pcolormesh plot
        x = self.elements['x']
        y = self.elements['y']
        w = self.elements['w']
        
        Nx = 0
        seenVals = []
        for val in x:
            
            if val not in seenVals:
                Nx += 1
                seenVals.append(val)
        
        Ny = 0
        seenVals = []
        for val in y:
            
            if val not in seenVals:
                Ny += 1
                seenVals.append(val)
                
        # xx = np.reshape(x, (Nx, Ny))
        # yy = np.reshape(y, (Nx, Ny))
        ww = np.reshape(w, (Nx, Ny))
        
        plt.figure()
        plt.pcolormesh(ww)
        plt.title('Element Weights')
        plt.grid()
        plt.show()
        
    def plotPattCut(self, phi, ax = None):
        #Given a specified phi, plots the pattern for -pi/2 < theta < pi/2
        
        #Generate pattern cut
        theta = np.linspace(-pi/2, pi/2, 1000)
        patt = self.arrayFactor(theta, phi)
        patt = linTodB( normal(patt) )
        
        if ax == None:
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            
            ax.plot(theta * 180/pi, patt)
            ax.set_xlabel('Theta')
            ax.grid(True)
            
            plt.show()
            
        else:
            
            ax.plot(theta * 180/pi, patt)
            
            plt.show()
            
        return ax
    
    
class Array_Config:
    
    def __init__(self, Nx: int, Ny: int, dx: float, dy: float, w: float = 1.0):
        """
        Parameters
        ----------
        Nx : Integer
             Number of element rows
        Ny : Integer
             Number of element columns
        dx : Float (usually between 0.5 and 1)
             Spacing in x-dimension between elements
        dy : Float (usually between 0.5 and 1)
             Spacing in y-dimension between elements
        w :  Float  (Right now)
             Constant weight for all elements of an array.  Defaults to 1.
             In development feature

        Returns
        -------
        Creates grids of points to be used as elements for the Array_2D class.

        """
        
        self.Nx = Nx
        self.Ny = Ny
        self.dx = dx
        self.dy = dy
        self.w  = w
        
        
    def rectArray(self):
        
        x_pos = np.arange(0, self.Nx * self.dx, self.dx)
        y_pos = np.arange(0, self.Ny * self.dy, self.dy)
        
        xx, yy = np.meshgrid(x_pos, y_pos)
        w = np.ones(xx.shape) * self.w   #TODO: Only true for constant weights, revise for other cases
        
        elements = {}
        elements["x"] = np.ndarray.flatten(xx)
        elements["y"] = np.ndarray.flatten(yy)
        elements["w"] = np.ndarray.flatten(w)
        
        self.elements = elements
        
        return self.elements
    
    
    def triangArray(self, odd = 0):
        
        elements = self.rectArray()
        
        for idx, x in enumerate(elements['x']):
            
            if idx % 2 == odd:
                
                elements['w'][idx] = 0
                
        elements['x'] = np.array([x_val for idx, x_val in enumerate(elements['x'])  if (elements['w'][idx] != 0)])
        elements['y'] = np.array([y_val for idx, y_val in enumerate(elements['y'])  if (elements['w'][idx] != 0)])
        elements['w'] = np.array([w_val for idx, w_val in enumerate(elements['w'])  if (elements['w'][idx] != 0)])
        
        self.elements = elements
    
        return self.elements
                
    def delElement(self, xpos, ypos, elements):
        
        for idx, x in enumerate(elements['x']):
            
            if np.abs(elements['x'][idx] - xpos) < 0.01 and np.abs(elements['y'][idx] - ypos) < 0.01:
                
                elements['x'] = np.array([x_val for jdx, x_val in enumerate(elements['x'])  if jdx != idx])
                elements['y'] = np.array([y_val for jdx, y_val in enumerate(elements['y'])  if jdx != idx])
                elements['w'] = np.array([w_val for jdx, w_val in enumerate(elements['w'])  if jdx != idx])
                
                return elements
            
    def delElements(self, xbounds, ybounds, elements):
        
        def checkBounds(val, bounds):
            
            if val <= bounds[1] and val >= bounds[0]:
                return True
            else:
                return False
        
        
        keepIdx = []
        for idx, x in enumerate(elements['x']):
            
            if checkBounds(elements['x'][idx], xbounds) and checkBounds(elements['y'][idx], ybounds):
                
                continue
            else:
                keepIdx.append(idx)
                
        elements['x'] = elements['x'][keepIdx]
        elements['y'] = elements['y'][keepIdx]
        elements['w'] = elements['w'][keepIdx]
        
        return elements
            
            
