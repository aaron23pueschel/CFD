'''This Python code is an automatically generated wrapper
for Fortran code made by 'fmodpy'. The original documentation
for the Fortran source code follows.


'''

import os
import ctypes
import platform
import numpy

# --------------------------------------------------------------------
#               CONFIGURATION
# 
_verbose = True
_fort_compiler = "gfortran"
_shared_object_name = "upwind." + platform.machine() + ".so"
_this_directory = os.path.dirname(os.path.abspath(__file__))
_path_to_lib = os.path.join(_this_directory, _shared_object_name)
_compile_options = ['-fPIC', '-shared', '-O3']
_ordered_dependencies = ['upwind.f95', 'upwind_c_wrapper.f90']
_symbol_files = []# 
# --------------------------------------------------------------------
#               AUTO-COMPILING
#
# Try to import the prerequisite symbols for the compiled code.
for _ in _symbol_files:
    _ = ctypes.CDLL(os.path.join(_this_directory, _), mode=ctypes.RTLD_GLOBAL)
# Try to import the existing object. If that fails, recompile and then try.
try:
    # Check to see if the source files have been modified and a recompilation is needed.
    if (max(max([0]+[os.path.getmtime(os.path.realpath(os.path.join(_this_directory,_))) for _ in _symbol_files]),
            max([0]+[os.path.getmtime(os.path.realpath(os.path.join(_this_directory,_))) for _ in _ordered_dependencies]))
        > os.path.getmtime(_path_to_lib)):
        print()
        print("WARNING: Recompiling because the modification time of a source file is newer than the library.", flush=True)
        print()
        if os.path.exists(_path_to_lib):
            os.remove(_path_to_lib)
        raise NotImplementedError(f"The newest library code has not been compiled.")
    # Import the library.
    clib = ctypes.CDLL(_path_to_lib)
except:
    # Remove the shared object if it exists, because it is faulty.
    if os.path.exists(_shared_object_name):
        os.remove(_shared_object_name)
    # Compile a new shared object.
    _command = [_fort_compiler] + _ordered_dependencies + _compile_options + ["-o", _shared_object_name]
    if _verbose:
        print("Running system command with arguments")
        print("  ", " ".join(_command))
    # Run the compilation command.
    import subprocess
    subprocess.check_call(_command, cwd=_this_directory)
    # Import the shared object file as a C library with ctypes.
    clib = ctypes.CDLL(_path_to_lib)
# --------------------------------------------------------------------


class upwind_module:
    ''''''

    
    # ----------------------------------------------
    # Wrapper for the Fortran subroutine VANLEER_FLUX
    
    def vanleer_flux(self, vleft, vright, nx, ny, f=None):
        ''''''
        
        # Setting up "vleft"
        if ((not issubclass(type(vleft), numpy.ndarray)) or
            (not numpy.asarray(vleft).flags.f_contiguous) or
            (not (vleft.dtype == numpy.dtype(ctypes.c_double)))):
            import warnings
            warnings.warn("The provided argument 'vleft' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
            vleft = numpy.asarray(vleft, dtype=ctypes.c_double, order='F')
        vleft_dim_1 = ctypes.c_long(vleft.shape[0])
        
        # Setting up "vright"
        if ((not issubclass(type(vright), numpy.ndarray)) or
            (not numpy.asarray(vright).flags.f_contiguous) or
            (not (vright.dtype == numpy.dtype(ctypes.c_double)))):
            import warnings
            warnings.warn("The provided argument 'vright' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
            vright = numpy.asarray(vright, dtype=ctypes.c_double, order='F')
        vright_dim_1 = ctypes.c_long(vright.shape[0])
        
        # Setting up "nx"
        if (type(nx) is not ctypes.c_double): nx = ctypes.c_double(nx)
        
        # Setting up "ny"
        if (type(ny) is not ctypes.c_double): ny = ctypes.c_double(ny)
        
        # Setting up "f"
        if (f is None):
            f = numpy.zeros(shape=(4), dtype=ctypes.c_double, order='F')
        elif ((not issubclass(type(f), numpy.ndarray)) or
              (not numpy.asarray(f).flags.f_contiguous) or
              (not (f.dtype == numpy.dtype(ctypes.c_double)))):
            import warnings
            warnings.warn("The provided argument 'f' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
            f = numpy.asarray(f, dtype=ctypes.c_double, order='F')
        f_dim_1 = ctypes.c_long(f.shape[0])
    
        # Call C-accessible Fortran wrapper.
        clib.c_vanleer_flux(ctypes.byref(vleft_dim_1), ctypes.c_void_p(vleft.ctypes.data), ctypes.byref(vright_dim_1), ctypes.c_void_p(vright.ctypes.data), ctypes.byref(nx), ctypes.byref(ny), ctypes.byref(f_dim_1), ctypes.c_void_p(f.ctypes.data))
    
        # Return final results, 'INTENT(OUT)' arguments only.
        return f

    
    # ----------------------------------------------
    # Wrapper for the Fortran subroutine ROE_FLUX
    
    def roe_flux(self, vleft, vright, nx, ny, f=None):
        ''''''
        
        # Setting up "vleft"
        if ((not issubclass(type(vleft), numpy.ndarray)) or
            (not numpy.asarray(vleft).flags.f_contiguous) or
            (not (vleft.dtype == numpy.dtype(ctypes.c_double)))):
            import warnings
            warnings.warn("The provided argument 'vleft' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
            vleft = numpy.asarray(vleft, dtype=ctypes.c_double, order='F')
        vleft_dim_1 = ctypes.c_long(vleft.shape[0])
        
        # Setting up "vright"
        if ((not issubclass(type(vright), numpy.ndarray)) or
            (not numpy.asarray(vright).flags.f_contiguous) or
            (not (vright.dtype == numpy.dtype(ctypes.c_double)))):
            import warnings
            warnings.warn("The provided argument 'vright' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
            vright = numpy.asarray(vright, dtype=ctypes.c_double, order='F')
        vright_dim_1 = ctypes.c_long(vright.shape[0])
        
        # Setting up "nx"
        if (type(nx) is not ctypes.c_double): nx = ctypes.c_double(nx)
        
        # Setting up "ny"
        if (type(ny) is not ctypes.c_double): ny = ctypes.c_double(ny)
        
        # Setting up "f"
        if (f is None):
            f = numpy.zeros(shape=(4), dtype=ctypes.c_double, order='F')
        elif ((not issubclass(type(f), numpy.ndarray)) or
              (not numpy.asarray(f).flags.f_contiguous) or
              (not (f.dtype == numpy.dtype(ctypes.c_double)))):
            import warnings
            warnings.warn("The provided argument 'f' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
            f = numpy.asarray(f, dtype=ctypes.c_double, order='F')
        f_dim_1 = ctypes.c_long(f.shape[0])
    
        # Call C-accessible Fortran wrapper.
        clib.c_roe_flux(ctypes.byref(vleft_dim_1), ctypes.c_void_p(vleft.ctypes.data), ctypes.byref(vright_dim_1), ctypes.c_void_p(vright.ctypes.data), ctypes.byref(nx), ctypes.byref(ny), ctypes.byref(f_dim_1), ctypes.c_void_p(f.ctypes.data))
    
        # Return final results, 'INTENT(OUT)' arguments only.
        return f

    
    # ----------------------------------------------
    # Wrapper for the Fortran subroutine COMPUTE_L_R_STATES_ETA
    
    def compute_l_r_states_eta(self, v, imax, jmax, vl_eta=None, vr_eta=None):
        ''''''
        
        # Setting up "v"
        if ((not issubclass(type(v), numpy.ndarray)) or
            (not numpy.asarray(v).flags.f_contiguous) or
            (not (v.dtype == numpy.dtype(ctypes.c_double)))):
            import warnings
            warnings.warn("The provided argument 'v' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
            v = numpy.asarray(v, dtype=ctypes.c_double, order='F')
        v_dim_1 = ctypes.c_long(v.shape[0])
        v_dim_2 = ctypes.c_long(v.shape[1])
        v_dim_3 = ctypes.c_long(v.shape[2])
        
        # Setting up "imax"
        if (type(imax) is not ctypes.c_int): imax = ctypes.c_int(imax)
        
        # Setting up "jmax"
        if (type(jmax) is not ctypes.c_int): jmax = ctypes.c_int(jmax)
        
        # Setting up "vl_eta"
        if (vl_eta is None):
            vl_eta = numpy.zeros(shape=(4, imax-1, jmax), dtype=ctypes.c_double, order='F')
        elif ((not issubclass(type(vl_eta), numpy.ndarray)) or
              (not numpy.asarray(vl_eta).flags.f_contiguous) or
              (not (vl_eta.dtype == numpy.dtype(ctypes.c_double)))):
            import warnings
            warnings.warn("The provided argument 'vl_eta' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
            vl_eta = numpy.asarray(vl_eta, dtype=ctypes.c_double, order='F')
        vl_eta_dim_1 = ctypes.c_long(vl_eta.shape[0])
        vl_eta_dim_2 = ctypes.c_long(vl_eta.shape[1])
        vl_eta_dim_3 = ctypes.c_long(vl_eta.shape[2])
        
        # Setting up "vr_eta"
        if (vr_eta is None):
            vr_eta = numpy.zeros(shape=(4, imax-1, jmax), dtype=ctypes.c_double, order='F')
        elif ((not issubclass(type(vr_eta), numpy.ndarray)) or
              (not numpy.asarray(vr_eta).flags.f_contiguous) or
              (not (vr_eta.dtype == numpy.dtype(ctypes.c_double)))):
            import warnings
            warnings.warn("The provided argument 'vr_eta' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
            vr_eta = numpy.asarray(vr_eta, dtype=ctypes.c_double, order='F')
        vr_eta_dim_1 = ctypes.c_long(vr_eta.shape[0])
        vr_eta_dim_2 = ctypes.c_long(vr_eta.shape[1])
        vr_eta_dim_3 = ctypes.c_long(vr_eta.shape[2])
    
        # Call C-accessible Fortran wrapper.
        clib.c_compute_l_r_states_eta(ctypes.byref(v_dim_1), ctypes.byref(v_dim_2), ctypes.byref(v_dim_3), ctypes.c_void_p(v.ctypes.data), ctypes.byref(imax), ctypes.byref(jmax), ctypes.byref(vl_eta_dim_1), ctypes.byref(vl_eta_dim_2), ctypes.byref(vl_eta_dim_3), ctypes.c_void_p(vl_eta.ctypes.data), ctypes.byref(vr_eta_dim_1), ctypes.byref(vr_eta_dim_2), ctypes.byref(vr_eta_dim_3), ctypes.c_void_p(vr_eta.ctypes.data))
    
        # Return final results, 'INTENT(OUT)' arguments only.
        return vl_eta, vr_eta

upwind_module = upwind_module()

