import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer

class C_FUNCTIONS:
    """
    Class to initialize all the functions in functions.c
    """
    def __init__(self):
        self.functions = {}
        try:
            self.funcs = ctypes.cdll.LoadLibrary("./c_functions.so")
        except:
            raise FileNotFoundError("Compile functions.c: See documentation")
        self.float_pointer_3d = ndpointer(dtype=np.float32,ndim=3, flags="C_CONTIGUOUS")
        self.float_pointer_2d = ndpointer(dtype=np.float32,ndim=2, flags="C_CONTIGUOUS")
        self.double_pointer = ndpointer(dtype=np.float32,ndim=1,flags="C_CONTIGUOUS")
    

    def FM(self):
        fm = self.funcs.FM
        fm.restype = ctypes.c_double
        fm.argtype = [ctypes.c_double,ctypes.c_double,ctypes.c_double]
        return fm

    def DfdM(self):
        fm = self.funcs.DfdM
        fm.restype = ctypes.c_double
        fm.argtype = [ctypes.c_double,ctypes.c_double,ctypes.c_double]
        return fm

    def newton(self):
        fm = self.funcs.newton
        fm.restype = ctypes.c_double
        fm.argtype = [ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double]
        return fm


    def T(gamma,M,T0):
        return T0/psi(gamma,M)

    def P(gamma,M,P0):
        return P0/(psi(gamma,M)**(gamma/(gamma-1)))
    def rho(P,R,T):
        return P/(R*T)
    def u(gamma,M,R,T):
        return M*np.sqrt(gamma*R*T)

def psi(gamma,M):
    return 1+((gamma-1)/2)*M**2

  