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
_shared_object_name = "flux_module." + platform.machine() + ".so"
_this_directory = os.path.dirname(os.path.abspath(__file__))
_path_to_lib = os.path.join(_this_directory, _shared_object_name)
_compile_options = ['-fPIC', '-shared', '-O3']
_ordered_dependencies = ['flux_module.f95', 'flux_module_c_wrapper.f90']
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


class flux_module:
    ''''''

    # Declare 'dp'
    def get_dp(self):
        dp = ctypes.c_int()
        clib.flux_module_get_dp(ctypes.byref(dp))
        return dp.value
    def set_dp(self, dp):
        raise(NotImplementedError('Module attributes with PARAMETER status cannot be set.'))
    dp = property(get_dp, set_dp)

    # Declare 'imax'
    def get_imax(self):
        imax = ctypes.c_int()
        clib.flux_module_get_imax(ctypes.byref(imax))
        return imax.value
    def set_imax(self, imax):
        raise(NotImplementedError('Module attributes with PARAMETER status cannot be set.'))
    imax = property(get_imax, set_imax)

    # Declare 'jmax'
    def get_jmax(self):
        jmax = ctypes.c_int()
        clib.flux_module_get_jmax(ctypes.byref(jmax))
        return jmax.value
    def set_jmax(self, jmax):
        raise(NotImplementedError('Module attributes with PARAMETER status cannot be set.'))
    jmax = property(get_jmax, set_jmax)

    # Declare 'zero'
    def get_zero(self):
        zero = ctypes.c_int()
        clib.flux_module_get_zero(ctypes.byref(zero))
        return zero.value
    def set_zero(self, zero):
        raise(NotImplementedError('Module attributes with PARAMETER status cannot be set.'))
    zero = property(get_zero, set_zero)

    # Declare 'f_xsi'
    def get_f_xsi(self):
        f_xsi_dim_1 = ctypes.c_long()
        f_xsi_dim_2 = ctypes.c_long()
        f_xsi_dim_3 = ctypes.c_long()
        f_xsi = ctypes.c_void_p()
        clib.flux_module_get_f_xsi(ctypes.byref(f_xsi_dim_1), ctypes.byref(f_xsi_dim_2), ctypes.byref(f_xsi_dim_3), ctypes.byref(f_xsi))
        f_xsi_size = (f_xsi_dim_1.value) * (f_xsi_dim_2.value) * (f_xsi_dim_3.value)
        if (f_xsi_size > 0):
            f_xsi = numpy.array(ctypes.cast(f_xsi, ctypes.POINTER(ctypes.c_double*f_xsi_size)).contents, copy=False)
        else:
            f_xsi = numpy.zeros((0,), dtype=ctypes.c_double, order='F')
        f_xsi = f_xsi.reshape(f_xsi_dim_3.value,f_xsi_dim_2.value,f_xsi_dim_1.value).T
        return f_xsi
    def set_f_xsi(self, f_xsi):
        if ((not issubclass(type(f_xsi), numpy.ndarray)) or
            (not numpy.asarray(f_xsi).flags.f_contiguous) or
            (not (f_xsi.dtype == numpy.dtype(ctypes.c_double)))):
            import warnings
            warnings.warn("The provided argument 'f_xsi' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
            f_xsi = numpy.asarray(f_xsi, dtype=ctypes.c_double, order='F')
        f_xsi_dim_1 = ctypes.c_long(f_xsi.shape[0])
        f_xsi_dim_2 = ctypes.c_long(f_xsi.shape[1])
        f_xsi_dim_3 = ctypes.c_long(f_xsi.shape[2])
        clib.flux_module_set_f_xsi(ctypes.byref(f_xsi_dim_1), ctypes.byref(f_xsi_dim_2), ctypes.byref(f_xsi_dim_3), ctypes.c_void_p(f_xsi.ctypes.data))
    f_xsi = property(get_f_xsi, set_f_xsi)

    # Declare 'f_eta'
    def get_f_eta(self):
        f_eta_dim_1 = ctypes.c_long()
        f_eta_dim_2 = ctypes.c_long()
        f_eta_dim_3 = ctypes.c_long()
        f_eta = ctypes.c_void_p()
        clib.flux_module_get_f_eta(ctypes.byref(f_eta_dim_1), ctypes.byref(f_eta_dim_2), ctypes.byref(f_eta_dim_3), ctypes.byref(f_eta))
        f_eta_size = (f_eta_dim_1.value) * (f_eta_dim_2.value) * (f_eta_dim_3.value)
        if (f_eta_size > 0):
            f_eta = numpy.array(ctypes.cast(f_eta, ctypes.POINTER(ctypes.c_double*f_eta_size)).contents, copy=False)
        else:
            f_eta = numpy.zeros((0,), dtype=ctypes.c_double, order='F')
        f_eta = f_eta.reshape(f_eta_dim_3.value,f_eta_dim_2.value,f_eta_dim_1.value).T
        return f_eta
    def set_f_eta(self, f_eta):
        if ((not issubclass(type(f_eta), numpy.ndarray)) or
            (not numpy.asarray(f_eta).flags.f_contiguous) or
            (not (f_eta.dtype == numpy.dtype(ctypes.c_double)))):
            import warnings
            warnings.warn("The provided argument 'f_eta' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
            f_eta = numpy.asarray(f_eta, dtype=ctypes.c_double, order='F')
        f_eta_dim_1 = ctypes.c_long(f_eta.shape[0])
        f_eta_dim_2 = ctypes.c_long(f_eta.shape[1])
        f_eta_dim_3 = ctypes.c_long(f_eta.shape[2])
        clib.flux_module_set_f_eta(ctypes.byref(f_eta_dim_1), ctypes.byref(f_eta_dim_2), ctypes.byref(f_eta_dim_3), ctypes.c_void_p(f_eta.ctypes.data))
    f_eta = property(get_f_eta, set_f_eta)

    # Declare 'vanleer'
    def get_vanleer(self):
        vanleer = ctypes.c_int()
        clib.flux_module_get_vanleer(ctypes.byref(vanleer))
        return vanleer.value
    def set_vanleer(self, vanleer):
        vanleer = ctypes.c_int(vanleer)
        clib.flux_module_set_vanleer(ctypes.byref(vanleer))
    vanleer = property(get_vanleer, set_vanleer)

    # Declare 'epsilon'
    def get_epsilon(self):
        epsilon = ctypes.c_double()
        clib.flux_module_get_epsilon(ctypes.byref(epsilon))
        return epsilon.value
    def set_epsilon(self, epsilon):
        raise(NotImplementedError('Module attributes with PARAMETER status cannot be set.'))
    epsilon = property(get_epsilon, set_epsilon)

    # Declare 'conv'
    def get_conv(self):
        conv = ctypes.c_double()
        clib.flux_module_get_conv(ctypes.byref(conv))
        return conv.value
    def set_conv(self, conv):
        conv = ctypes.c_double(conv)
        clib.flux_module_set_conv(ctypes.byref(conv))
    conv = property(get_conv, set_conv)

    # Declare 'clip_count_rho'
    def get_clip_count_rho(self):
        clip_count_rho = ctypes.c_int()
        clib.flux_module_get_clip_count_rho(ctypes.byref(clip_count_rho))
        return clip_count_rho.value
    def set_clip_count_rho(self, clip_count_rho):
        clip_count_rho = ctypes.c_int(clip_count_rho)
        clib.flux_module_set_clip_count_rho(ctypes.byref(clip_count_rho))
    clip_count_rho = property(get_clip_count_rho, set_clip_count_rho)

    # Declare 'clip_count_p'
    def get_clip_count_p(self):
        clip_count_p = ctypes.c_int()
        clib.flux_module_get_clip_count_p(ctypes.byref(clip_count_p))
        return clip_count_p.value
    def set_clip_count_p(self, clip_count_p):
        clip_count_p = ctypes.c_int(clip_count_p)
        clib.flux_module_set_clip_count_p(ctypes.byref(clip_count_p))
    clip_count_p = property(get_clip_count_p, set_clip_count_p)

    # Declare 'total_flux_calls'
    def get_total_flux_calls(self):
        total_flux_calls = ctypes.c_int()
        clib.flux_module_get_total_flux_calls(ctypes.byref(total_flux_calls))
        return total_flux_calls.value
    def set_total_flux_calls(self, total_flux_calls):
        total_flux_calls = ctypes.c_int(total_flux_calls)
        clib.flux_module_set_total_flux_calls(ctypes.byref(total_flux_calls))
    total_flux_calls = property(get_total_flux_calls, set_total_flux_calls)

    # Declare 'freeze_limiters'
    def get_freeze_limiters(self):
        freeze_limiters = ctypes.c_int()
        clib.flux_module_get_freeze_limiters(ctypes.byref(freeze_limiters))
        return freeze_limiters.value
    def set_freeze_limiters(self, freeze_limiters):
        freeze_limiters = ctypes.c_int(freeze_limiters)
        clib.flux_module_set_freeze_limiters(ctypes.byref(freeze_limiters))
    freeze_limiters = property(get_freeze_limiters, set_freeze_limiters)

    # Declare 'psi_p_xsi_frozen'
    def get_psi_p_xsi_frozen(self):
        psi_p_xsi_frozen_dim_1 = ctypes.c_long()
        psi_p_xsi_frozen_dim_2 = ctypes.c_long()
        psi_p_xsi_frozen_dim_3 = ctypes.c_long()
        psi_p_xsi_frozen = ctypes.c_void_p()
        clib.flux_module_get_psi_p_xsi_frozen(ctypes.byref(psi_p_xsi_frozen_dim_1), ctypes.byref(psi_p_xsi_frozen_dim_2), ctypes.byref(psi_p_xsi_frozen_dim_3), ctypes.byref(psi_p_xsi_frozen))
        psi_p_xsi_frozen_size = (psi_p_xsi_frozen_dim_1.value) * (psi_p_xsi_frozen_dim_2.value) * (psi_p_xsi_frozen_dim_3.value)
        if (psi_p_xsi_frozen_size > 0):
            psi_p_xsi_frozen = numpy.array(ctypes.cast(psi_p_xsi_frozen, ctypes.POINTER(ctypes.c_double*psi_p_xsi_frozen_size)).contents, copy=False)
        else:
            psi_p_xsi_frozen = numpy.zeros((0,), dtype=ctypes.c_double, order='F')
        psi_p_xsi_frozen = psi_p_xsi_frozen.reshape(psi_p_xsi_frozen_dim_3.value,psi_p_xsi_frozen_dim_2.value,psi_p_xsi_frozen_dim_1.value).T
        return psi_p_xsi_frozen
    def set_psi_p_xsi_frozen(self, psi_p_xsi_frozen):
        if ((not issubclass(type(psi_p_xsi_frozen), numpy.ndarray)) or
            (not numpy.asarray(psi_p_xsi_frozen).flags.f_contiguous) or
            (not (psi_p_xsi_frozen.dtype == numpy.dtype(ctypes.c_double)))):
            import warnings
            warnings.warn("The provided argument 'psi_p_xsi_frozen' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
            psi_p_xsi_frozen = numpy.asarray(psi_p_xsi_frozen, dtype=ctypes.c_double, order='F')
        psi_p_xsi_frozen_dim_1 = ctypes.c_long(psi_p_xsi_frozen.shape[0])
        psi_p_xsi_frozen_dim_2 = ctypes.c_long(psi_p_xsi_frozen.shape[1])
        psi_p_xsi_frozen_dim_3 = ctypes.c_long(psi_p_xsi_frozen.shape[2])
        clib.flux_module_set_psi_p_xsi_frozen(ctypes.byref(psi_p_xsi_frozen_dim_1), ctypes.byref(psi_p_xsi_frozen_dim_2), ctypes.byref(psi_p_xsi_frozen_dim_3), ctypes.c_void_p(psi_p_xsi_frozen.ctypes.data))
    psi_p_xsi_frozen = property(get_psi_p_xsi_frozen, set_psi_p_xsi_frozen)

    # Declare 'psi_m_xsi_frozen'
    def get_psi_m_xsi_frozen(self):
        psi_m_xsi_frozen_dim_1 = ctypes.c_long()
        psi_m_xsi_frozen_dim_2 = ctypes.c_long()
        psi_m_xsi_frozen_dim_3 = ctypes.c_long()
        psi_m_xsi_frozen = ctypes.c_void_p()
        clib.flux_module_get_psi_m_xsi_frozen(ctypes.byref(psi_m_xsi_frozen_dim_1), ctypes.byref(psi_m_xsi_frozen_dim_2), ctypes.byref(psi_m_xsi_frozen_dim_3), ctypes.byref(psi_m_xsi_frozen))
        psi_m_xsi_frozen_size = (psi_m_xsi_frozen_dim_1.value) * (psi_m_xsi_frozen_dim_2.value) * (psi_m_xsi_frozen_dim_3.value)
        if (psi_m_xsi_frozen_size > 0):
            psi_m_xsi_frozen = numpy.array(ctypes.cast(psi_m_xsi_frozen, ctypes.POINTER(ctypes.c_double*psi_m_xsi_frozen_size)).contents, copy=False)
        else:
            psi_m_xsi_frozen = numpy.zeros((0,), dtype=ctypes.c_double, order='F')
        psi_m_xsi_frozen = psi_m_xsi_frozen.reshape(psi_m_xsi_frozen_dim_3.value,psi_m_xsi_frozen_dim_2.value,psi_m_xsi_frozen_dim_1.value).T
        return psi_m_xsi_frozen
    def set_psi_m_xsi_frozen(self, psi_m_xsi_frozen):
        if ((not issubclass(type(psi_m_xsi_frozen), numpy.ndarray)) or
            (not numpy.asarray(psi_m_xsi_frozen).flags.f_contiguous) or
            (not (psi_m_xsi_frozen.dtype == numpy.dtype(ctypes.c_double)))):
            import warnings
            warnings.warn("The provided argument 'psi_m_xsi_frozen' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
            psi_m_xsi_frozen = numpy.asarray(psi_m_xsi_frozen, dtype=ctypes.c_double, order='F')
        psi_m_xsi_frozen_dim_1 = ctypes.c_long(psi_m_xsi_frozen.shape[0])
        psi_m_xsi_frozen_dim_2 = ctypes.c_long(psi_m_xsi_frozen.shape[1])
        psi_m_xsi_frozen_dim_3 = ctypes.c_long(psi_m_xsi_frozen.shape[2])
        clib.flux_module_set_psi_m_xsi_frozen(ctypes.byref(psi_m_xsi_frozen_dim_1), ctypes.byref(psi_m_xsi_frozen_dim_2), ctypes.byref(psi_m_xsi_frozen_dim_3), ctypes.c_void_p(psi_m_xsi_frozen.ctypes.data))
    psi_m_xsi_frozen = property(get_psi_m_xsi_frozen, set_psi_m_xsi_frozen)

    # Declare 'psi_p_eta_frozen'
    def get_psi_p_eta_frozen(self):
        psi_p_eta_frozen_dim_1 = ctypes.c_long()
        psi_p_eta_frozen_dim_2 = ctypes.c_long()
        psi_p_eta_frozen_dim_3 = ctypes.c_long()
        psi_p_eta_frozen = ctypes.c_void_p()
        clib.flux_module_get_psi_p_eta_frozen(ctypes.byref(psi_p_eta_frozen_dim_1), ctypes.byref(psi_p_eta_frozen_dim_2), ctypes.byref(psi_p_eta_frozen_dim_3), ctypes.byref(psi_p_eta_frozen))
        psi_p_eta_frozen_size = (psi_p_eta_frozen_dim_1.value) * (psi_p_eta_frozen_dim_2.value) * (psi_p_eta_frozen_dim_3.value)
        if (psi_p_eta_frozen_size > 0):
            psi_p_eta_frozen = numpy.array(ctypes.cast(psi_p_eta_frozen, ctypes.POINTER(ctypes.c_double*psi_p_eta_frozen_size)).contents, copy=False)
        else:
            psi_p_eta_frozen = numpy.zeros((0,), dtype=ctypes.c_double, order='F')
        psi_p_eta_frozen = psi_p_eta_frozen.reshape(psi_p_eta_frozen_dim_3.value,psi_p_eta_frozen_dim_2.value,psi_p_eta_frozen_dim_1.value).T
        return psi_p_eta_frozen
    def set_psi_p_eta_frozen(self, psi_p_eta_frozen):
        if ((not issubclass(type(psi_p_eta_frozen), numpy.ndarray)) or
            (not numpy.asarray(psi_p_eta_frozen).flags.f_contiguous) or
            (not (psi_p_eta_frozen.dtype == numpy.dtype(ctypes.c_double)))):
            import warnings
            warnings.warn("The provided argument 'psi_p_eta_frozen' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
            psi_p_eta_frozen = numpy.asarray(psi_p_eta_frozen, dtype=ctypes.c_double, order='F')
        psi_p_eta_frozen_dim_1 = ctypes.c_long(psi_p_eta_frozen.shape[0])
        psi_p_eta_frozen_dim_2 = ctypes.c_long(psi_p_eta_frozen.shape[1])
        psi_p_eta_frozen_dim_3 = ctypes.c_long(psi_p_eta_frozen.shape[2])
        clib.flux_module_set_psi_p_eta_frozen(ctypes.byref(psi_p_eta_frozen_dim_1), ctypes.byref(psi_p_eta_frozen_dim_2), ctypes.byref(psi_p_eta_frozen_dim_3), ctypes.c_void_p(psi_p_eta_frozen.ctypes.data))
    psi_p_eta_frozen = property(get_psi_p_eta_frozen, set_psi_p_eta_frozen)

    # Declare 'psi_m_eta_frozen'
    def get_psi_m_eta_frozen(self):
        psi_m_eta_frozen_dim_1 = ctypes.c_long()
        psi_m_eta_frozen_dim_2 = ctypes.c_long()
        psi_m_eta_frozen_dim_3 = ctypes.c_long()
        psi_m_eta_frozen = ctypes.c_void_p()
        clib.flux_module_get_psi_m_eta_frozen(ctypes.byref(psi_m_eta_frozen_dim_1), ctypes.byref(psi_m_eta_frozen_dim_2), ctypes.byref(psi_m_eta_frozen_dim_3), ctypes.byref(psi_m_eta_frozen))
        psi_m_eta_frozen_size = (psi_m_eta_frozen_dim_1.value) * (psi_m_eta_frozen_dim_2.value) * (psi_m_eta_frozen_dim_3.value)
        if (psi_m_eta_frozen_size > 0):
            psi_m_eta_frozen = numpy.array(ctypes.cast(psi_m_eta_frozen, ctypes.POINTER(ctypes.c_double*psi_m_eta_frozen_size)).contents, copy=False)
        else:
            psi_m_eta_frozen = numpy.zeros((0,), dtype=ctypes.c_double, order='F')
        psi_m_eta_frozen = psi_m_eta_frozen.reshape(psi_m_eta_frozen_dim_3.value,psi_m_eta_frozen_dim_2.value,psi_m_eta_frozen_dim_1.value).T
        return psi_m_eta_frozen
    def set_psi_m_eta_frozen(self, psi_m_eta_frozen):
        if ((not issubclass(type(psi_m_eta_frozen), numpy.ndarray)) or
            (not numpy.asarray(psi_m_eta_frozen).flags.f_contiguous) or
            (not (psi_m_eta_frozen.dtype == numpy.dtype(ctypes.c_double)))):
            import warnings
            warnings.warn("The provided argument 'psi_m_eta_frozen' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
            psi_m_eta_frozen = numpy.asarray(psi_m_eta_frozen, dtype=ctypes.c_double, order='F')
        psi_m_eta_frozen_dim_1 = ctypes.c_long(psi_m_eta_frozen.shape[0])
        psi_m_eta_frozen_dim_2 = ctypes.c_long(psi_m_eta_frozen.shape[1])
        psi_m_eta_frozen_dim_3 = ctypes.c_long(psi_m_eta_frozen.shape[2])
        clib.flux_module_set_psi_m_eta_frozen(ctypes.byref(psi_m_eta_frozen_dim_1), ctypes.byref(psi_m_eta_frozen_dim_2), ctypes.byref(psi_m_eta_frozen_dim_3), ctypes.c_void_p(psi_m_eta_frozen.ctypes.data))
    psi_m_eta_frozen = property(get_psi_m_eta_frozen, set_psi_m_eta_frozen)

    # Declare 'freeze_tol'
    def get_freeze_tol(self):
        freeze_tol = ctypes.c_double()
        clib.flux_module_get_freeze_tol(ctypes.byref(freeze_tol))
        return freeze_tol.value
    def set_freeze_tol(self, freeze_tol):
        raise(NotImplementedError('Module attributes with PARAMETER status cannot be set.'))
    freeze_tol = property(get_freeze_tol, set_freeze_tol)

    # Declare 'one'
    def get_one(self):
        one = ctypes.c_double()
        clib.flux_module_get_one(ctypes.byref(one))
        return one.value
    def set_one(self, one):
        raise(NotImplementedError('Module attributes with PARAMETER status cannot be set.'))
    one = property(get_one, set_one)

    # Declare 'two'
    def get_two(self):
        two = ctypes.c_double()
        clib.flux_module_get_two(ctypes.byref(two))
        return two.value
    def set_two(self, two):
        raise(NotImplementedError('Module attributes with PARAMETER status cannot be set.'))
    two = property(get_two, set_two)

    # Declare 'quarter'
    def get_quarter(self):
        quarter = ctypes.c_double()
        clib.flux_module_get_quarter(ctypes.byref(quarter))
        return quarter.value
    def set_quarter(self, quarter):
        raise(NotImplementedError('Module attributes with PARAMETER status cannot be set.'))
    quarter = property(get_quarter, set_quarter)

    # Declare 'half'
    def get_half(self):
        half = ctypes.c_double()
        clib.flux_module_get_half(ctypes.byref(half))
        return half.value
    def set_half(self, half):
        raise(NotImplementedError('Module attributes with PARAMETER status cannot be set.'))
    half = property(get_half, set_half)

    # Declare 'gamma'
    def get_gamma(self):
        gamma = ctypes.c_double()
        clib.flux_module_get_gamma(ctypes.byref(gamma))
        return gamma.value
    def set_gamma(self, gamma):
        raise(NotImplementedError('Module attributes with PARAMETER status cannot be set.'))
    gamma = property(get_gamma, set_gamma)

    # Declare 'four'
    def get_four(self):
        four = ctypes.c_double()
        clib.flux_module_get_four(ctypes.byref(four))
        return four.value
    def set_four(self, four):
        raise(NotImplementedError('Module attributes with PARAMETER status cannot be set.'))
    four = property(get_four, set_four)

    
    # ----------------------------------------------
    # Wrapper for the Fortran subroutine COMPUTE_LIMITER_XSI
    
    def compute_limiter_xsi(self, v, psi_p_xsi=None, psi_m_xsi=None):
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
        
        # Setting up "psi_p_xsi"
        if (psi_p_xsi is None):
            psi_p_xsi = numpy.zeros(shape=(4, 1:self.imax, 1:self.jmax-1), dtype=ctypes.c_double, order='F')
        elif ((not issubclass(type(psi_p_xsi), numpy.ndarray)) or
              (not numpy.asarray(psi_p_xsi).flags.f_contiguous) or
              (not (psi_p_xsi.dtype == numpy.dtype(ctypes.c_double)))):
            import warnings
            warnings.warn("The provided argument 'psi_p_xsi' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
            psi_p_xsi = numpy.asarray(psi_p_xsi, dtype=ctypes.c_double, order='F')
        psi_p_xsi_dim_1 = ctypes.c_long(psi_p_xsi.shape[0])
        psi_p_xsi_dim_2 = ctypes.c_long(psi_p_xsi.shape[1])
        psi_p_xsi_dim_3 = ctypes.c_long(psi_p_xsi.shape[2])
        
        # Setting up "psi_m_xsi"
        if (psi_m_xsi is None):
            psi_m_xsi = numpy.zeros(shape=(4, 1:self.imax, 1:self.jmax-1), dtype=ctypes.c_double, order='F')
        elif ((not issubclass(type(psi_m_xsi), numpy.ndarray)) or
              (not numpy.asarray(psi_m_xsi).flags.f_contiguous) or
              (not (psi_m_xsi.dtype == numpy.dtype(ctypes.c_double)))):
            import warnings
            warnings.warn("The provided argument 'psi_m_xsi' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
            psi_m_xsi = numpy.asarray(psi_m_xsi, dtype=ctypes.c_double, order='F')
        psi_m_xsi_dim_1 = ctypes.c_long(psi_m_xsi.shape[0])
        psi_m_xsi_dim_2 = ctypes.c_long(psi_m_xsi.shape[1])
        psi_m_xsi_dim_3 = ctypes.c_long(psi_m_xsi.shape[2])
    
        # Call C-accessible Fortran wrapper.
        clib.c_compute_limiter_xsi(ctypes.byref(v_dim_1), ctypes.byref(v_dim_2), ctypes.byref(v_dim_3), ctypes.c_void_p(v.ctypes.data), ctypes.byref(psi_p_xsi_dim_1), ctypes.byref(psi_p_xsi_dim_2), ctypes.byref(psi_p_xsi_dim_3), ctypes.c_void_p(psi_p_xsi.ctypes.data), ctypes.byref(psi_m_xsi_dim_1), ctypes.byref(psi_m_xsi_dim_2), ctypes.byref(psi_m_xsi_dim_3), ctypes.c_void_p(psi_m_xsi.ctypes.data))
    
        # Return final results, 'INTENT(OUT)' arguments only.
        return psi_p_xsi, psi_m_xsi

    
    # ----------------------------------------------
    # Wrapper for the Fortran subroutine COMPUTE_L_R_STATES_XSI
    
    def compute_l_r_states_xsi(self, v, vl_xsi=None, vr_xsi=None):
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
        
        # Setting up "vl_xsi"
        if (vl_xsi is None):
            vl_xsi = numpy.zeros(shape=(4, 1:self.imax, 1:self.jmax-1), dtype=ctypes.c_double, order='F')
        elif ((not issubclass(type(vl_xsi), numpy.ndarray)) or
              (not numpy.asarray(vl_xsi).flags.f_contiguous) or
              (not (vl_xsi.dtype == numpy.dtype(ctypes.c_double)))):
            import warnings
            warnings.warn("The provided argument 'vl_xsi' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
            vl_xsi = numpy.asarray(vl_xsi, dtype=ctypes.c_double, order='F')
        vl_xsi_dim_1 = ctypes.c_long(vl_xsi.shape[0])
        vl_xsi_dim_2 = ctypes.c_long(vl_xsi.shape[1])
        vl_xsi_dim_3 = ctypes.c_long(vl_xsi.shape[2])
        
        # Setting up "vr_xsi"
        if (vr_xsi is None):
            vr_xsi = numpy.zeros(shape=(4, 1:self.imax, 1:self.jmax-1), dtype=ctypes.c_double, order='F')
        elif ((not issubclass(type(vr_xsi), numpy.ndarray)) or
              (not numpy.asarray(vr_xsi).flags.f_contiguous) or
              (not (vr_xsi.dtype == numpy.dtype(ctypes.c_double)))):
            import warnings
            warnings.warn("The provided argument 'vr_xsi' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
            vr_xsi = numpy.asarray(vr_xsi, dtype=ctypes.c_double, order='F')
        vr_xsi_dim_1 = ctypes.c_long(vr_xsi.shape[0])
        vr_xsi_dim_2 = ctypes.c_long(vr_xsi.shape[1])
        vr_xsi_dim_3 = ctypes.c_long(vr_xsi.shape[2])
    
        # Call C-accessible Fortran wrapper.
        clib.c_compute_l_r_states_xsi(ctypes.byref(v_dim_1), ctypes.byref(v_dim_2), ctypes.byref(v_dim_3), ctypes.c_void_p(v.ctypes.data), ctypes.byref(vl_xsi_dim_1), ctypes.byref(vl_xsi_dim_2), ctypes.byref(vl_xsi_dim_3), ctypes.c_void_p(vl_xsi.ctypes.data), ctypes.byref(vr_xsi_dim_1), ctypes.byref(vr_xsi_dim_2), ctypes.byref(vr_xsi_dim_3), ctypes.c_void_p(vr_xsi.ctypes.data))
    
        # Return final results, 'INTENT(OUT)' arguments only.
        return vl_xsi, vr_xsi

    
    # ----------------------------------------------
    # Wrapper for the Fortran subroutine COMPUTE_LIMITER_ETA
    
    def compute_limiter_eta(self, v, psi_p_eta=None, psi_m_eta=None):
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
        
        # Setting up "psi_p_eta"
        if (psi_p_eta is None):
            psi_p_eta = numpy.zeros(shape=(4, 1:self.imax-1, 1:self.jmax), dtype=ctypes.c_double, order='F')
        elif ((not issubclass(type(psi_p_eta), numpy.ndarray)) or
              (not numpy.asarray(psi_p_eta).flags.f_contiguous) or
              (not (psi_p_eta.dtype == numpy.dtype(ctypes.c_double)))):
            import warnings
            warnings.warn("The provided argument 'psi_p_eta' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
            psi_p_eta = numpy.asarray(psi_p_eta, dtype=ctypes.c_double, order='F')
        psi_p_eta_dim_1 = ctypes.c_long(psi_p_eta.shape[0])
        psi_p_eta_dim_2 = ctypes.c_long(psi_p_eta.shape[1])
        psi_p_eta_dim_3 = ctypes.c_long(psi_p_eta.shape[2])
        
        # Setting up "psi_m_eta"
        if (psi_m_eta is None):
            psi_m_eta = numpy.zeros(shape=(4, 1:self.imax-1, 1:self.jmax), dtype=ctypes.c_double, order='F')
        elif ((not issubclass(type(psi_m_eta), numpy.ndarray)) or
              (not numpy.asarray(psi_m_eta).flags.f_contiguous) or
              (not (psi_m_eta.dtype == numpy.dtype(ctypes.c_double)))):
            import warnings
            warnings.warn("The provided argument 'psi_m_eta' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
            psi_m_eta = numpy.asarray(psi_m_eta, dtype=ctypes.c_double, order='F')
        psi_m_eta_dim_1 = ctypes.c_long(psi_m_eta.shape[0])
        psi_m_eta_dim_2 = ctypes.c_long(psi_m_eta.shape[1])
        psi_m_eta_dim_3 = ctypes.c_long(psi_m_eta.shape[2])
    
        # Call C-accessible Fortran wrapper.
        clib.c_compute_limiter_eta(ctypes.byref(v_dim_1), ctypes.byref(v_dim_2), ctypes.byref(v_dim_3), ctypes.c_void_p(v.ctypes.data), ctypes.byref(psi_p_eta_dim_1), ctypes.byref(psi_p_eta_dim_2), ctypes.byref(psi_p_eta_dim_3), ctypes.c_void_p(psi_p_eta.ctypes.data), ctypes.byref(psi_m_eta_dim_1), ctypes.byref(psi_m_eta_dim_2), ctypes.byref(psi_m_eta_dim_3), ctypes.c_void_p(psi_m_eta.ctypes.data))
    
        # Return final results, 'INTENT(OUT)' arguments only.
        return psi_p_eta, psi_m_eta

    
    # ----------------------------------------------
    # Wrapper for the Fortran subroutine COMPUTE_L_R_STATES_ETA
    
    def compute_l_r_states_eta(self, v, vl_eta=None, vr_eta=None):
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
        
        # Setting up "vl_eta"
        if (vl_eta is None):
            vl_eta = numpy.zeros(shape=(4, 1:self.imax-1, 1:self.jmax), dtype=ctypes.c_double, order='F')
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
            vr_eta = numpy.zeros(shape=(4, 1:self.imax-1, 1:self.jmax), dtype=ctypes.c_double, order='F')
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
        clib.c_compute_l_r_states_eta(ctypes.byref(v_dim_1), ctypes.byref(v_dim_2), ctypes.byref(v_dim_3), ctypes.c_void_p(v.ctypes.data), ctypes.byref(vl_eta_dim_1), ctypes.byref(vl_eta_dim_2), ctypes.byref(vl_eta_dim_3), ctypes.c_void_p(vl_eta.ctypes.data), ctypes.byref(vr_eta_dim_1), ctypes.byref(vr_eta_dim_2), ctypes.byref(vr_eta_dim_3), ctypes.c_void_p(vr_eta.ctypes.data))
    
        # Return final results, 'INTENT(OUT)' arguments only.
        return vl_eta, vr_eta

    
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
    # Wrapper for the Fortran subroutine COMPUTE_FLUX_XSI
    
    def compute_flux_xsi(self, vanleer, v, nhat_xsi, f_xsi=None):
        ''''''
        
        # Setting up "vanleer"
        if (type(vanleer) is not ctypes.c_int): vanleer = ctypes.c_int(vanleer)
        
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
        
        # Setting up "nhat_xsi"
        if ((not issubclass(type(nhat_xsi), numpy.ndarray)) or
            (not numpy.asarray(nhat_xsi).flags.f_contiguous) or
            (not (nhat_xsi.dtype == numpy.dtype(ctypes.c_double)))):
            import warnings
            warnings.warn("The provided argument 'nhat_xsi' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
            nhat_xsi = numpy.asarray(nhat_xsi, dtype=ctypes.c_double, order='F')
        nhat_xsi_dim_1 = ctypes.c_long(nhat_xsi.shape[0])
        nhat_xsi_dim_2 = ctypes.c_long(nhat_xsi.shape[1])
        nhat_xsi_dim_3 = ctypes.c_long(nhat_xsi.shape[2])
        
        # Setting up "f_xsi"
        if (f_xsi is None):
            f_xsi = numpy.zeros(shape=(4, 1:self.imax, 1:self.jmax-1), dtype=ctypes.c_double, order='F')
        elif ((not issubclass(type(f_xsi), numpy.ndarray)) or
              (not numpy.asarray(f_xsi).flags.f_contiguous) or
              (not (f_xsi.dtype == numpy.dtype(ctypes.c_double)))):
            import warnings
            warnings.warn("The provided argument 'f_xsi' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
            f_xsi = numpy.asarray(f_xsi, dtype=ctypes.c_double, order='F')
        f_xsi_dim_1 = ctypes.c_long(f_xsi.shape[0])
        f_xsi_dim_2 = ctypes.c_long(f_xsi.shape[1])
        f_xsi_dim_3 = ctypes.c_long(f_xsi.shape[2])
    
        # Call C-accessible Fortran wrapper.
        clib.c_compute_flux_xsi(ctypes.byref(vanleer), ctypes.byref(v_dim_1), ctypes.byref(v_dim_2), ctypes.byref(v_dim_3), ctypes.c_void_p(v.ctypes.data), ctypes.byref(nhat_xsi_dim_1), ctypes.byref(nhat_xsi_dim_2), ctypes.byref(nhat_xsi_dim_3), ctypes.c_void_p(nhat_xsi.ctypes.data), ctypes.byref(f_xsi_dim_1), ctypes.byref(f_xsi_dim_2), ctypes.byref(f_xsi_dim_3), ctypes.c_void_p(f_xsi.ctypes.data))
    
        # Return final results, 'INTENT(OUT)' arguments only.
        return f_xsi

    
    # ----------------------------------------------
    # Wrapper for the Fortran subroutine COMPUTE_FLUX_ETA
    
    def compute_flux_eta(self, vanleer, v, nhat_eta, f_eta=None):
        ''''''
        
        # Setting up "vanleer"
        if (type(vanleer) is not ctypes.c_int): vanleer = ctypes.c_int(vanleer)
        
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
        
        # Setting up "nhat_eta"
        if ((not issubclass(type(nhat_eta), numpy.ndarray)) or
            (not numpy.asarray(nhat_eta).flags.f_contiguous) or
            (not (nhat_eta.dtype == numpy.dtype(ctypes.c_double)))):
            import warnings
            warnings.warn("The provided argument 'nhat_eta' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
            nhat_eta = numpy.asarray(nhat_eta, dtype=ctypes.c_double, order='F')
        nhat_eta_dim_1 = ctypes.c_long(nhat_eta.shape[0])
        nhat_eta_dim_2 = ctypes.c_long(nhat_eta.shape[1])
        nhat_eta_dim_3 = ctypes.c_long(nhat_eta.shape[2])
        
        # Setting up "f_eta"
        if (f_eta is None):
            f_eta = numpy.zeros(shape=(4, 1:self.imax-1, 1:self.jmax), dtype=ctypes.c_double, order='F')
        elif ((not issubclass(type(f_eta), numpy.ndarray)) or
              (not numpy.asarray(f_eta).flags.f_contiguous) or
              (not (f_eta.dtype == numpy.dtype(ctypes.c_double)))):
            import warnings
            warnings.warn("The provided argument 'f_eta' was not an f_contiguous NumPy array of type 'ctypes.c_double' (or equivalent). Automatically converting (probably creating a full copy).")
            f_eta = numpy.asarray(f_eta, dtype=ctypes.c_double, order='F')
        f_eta_dim_1 = ctypes.c_long(f_eta.shape[0])
        f_eta_dim_2 = ctypes.c_long(f_eta.shape[1])
        f_eta_dim_3 = ctypes.c_long(f_eta.shape[2])
    
        # Call C-accessible Fortran wrapper.
        clib.c_compute_flux_eta(ctypes.byref(vanleer), ctypes.byref(v_dim_1), ctypes.byref(v_dim_2), ctypes.byref(v_dim_3), ctypes.c_void_p(v.ctypes.data), ctypes.byref(nhat_eta_dim_1), ctypes.byref(nhat_eta_dim_2), ctypes.byref(nhat_eta_dim_3), ctypes.c_void_p(nhat_eta.ctypes.data), ctypes.byref(f_eta_dim_1), ctypes.byref(f_eta_dim_2), ctypes.byref(f_eta_dim_3), ctypes.c_void_p(f_eta.ctypes.data))
    
        # Return final results, 'INTENT(OUT)' arguments only.
        return f_eta

flux_module = flux_module()

