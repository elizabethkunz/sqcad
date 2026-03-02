from __future__ import annotations
import numpy as np
# from core.plots import plot_lom_vs_data_re_im
import skrf as rf
from skrf import Circuit
from skrf.media import CPW
from .utils import *
from scipy.signal import find_peaks

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Sequence

import numpy as np
import skrf as rf
from scipy.optimize import least_squares


#Model 1: Foster Synthesis

def foster_synthesis(freq, w = 11.7e-6, s = 5.1e-6, t=0,h=500e-6,rho=1e-19,ep_r = 11.45, has_metal_backside=True, tand=0, d=0.007):
    
    '''
    Perform Foster synthesis to extract LOM parameters from a CPW resonator.
    Steps:
        1. Create a CPW resonator network and compute its S-parameters.
        2. Identify the resonance frequency f0 from the S-parameters.
        3. Compute the input admittance Yin and susceptance B = Im(Yin) at port 1.
        4. Evaluate dB/dw at the resonance frequency w0 = 2*pi*f0.
        5. Construct the LOM network using the extracted L_eq and C_eq.
    
    '''


    cpw = CPW(freq, w = w,   
                    s = s, #spacing = 5.1um
                    t=t, #thickness = 200nm
                    h=h, #substrate height = 525um
                    rho=rho, #closest to 0 this thing goes
                    ep_r = ep_r, #ultracold silicon
                    has_metal_backside=has_metal_backside,
                    tand=tand,)
    
    line = cpw.line(d=d, unit='m', name='cpw line')
    port=rf.Circuit.Port(freq, name="P1", z0=500)
    port2=rf.Circuit.Port(freq, name="P2", z0=500)

    cnx2 = [[(port, 0), (line, 0)],
            [(line, 1), (port2, 0)]]

    ckt = rf.Circuit(cnx2, name="Port_Resonator")
    ntw = ckt.network
    ntw.plot_s_db()
    f = ntw.frequency.f
    S22 = ntw.s[:, 1, 1] # it's actually s22; s11 is [:, 0, 0] 
    #min_index= np.argmin(S22)
    peaks, _ = find_peaks(S22)
    f0s = f[peaks]
    print(f0s)
    Y = ntw.y

    Y11 = Y[:,0,0]
    Y12 = Y[:,0,1]
    Y21 = Y[:,1,0]
    Y22 = Y[:,1,1]

    #Load admittance, Z0 = 500 for both ports so:
    YL = 1 / 500

    # Yin(w)
    Yin = Y11 - (Y12 * Y21) / (Y22 + YL)

    B = np.imag(Yin) 

    w0 = 2 * np.pi * f0s[0]
    w = 2*np.pi*ntw.frequency.f

     # slope dB/dw using finite differences
    dB_dw = np.gradient(B, w)

    # index closest to desired w0
    i0 = np.argmin(np.abs(w - w0))

    slope_at_w0 = dB_dw[i0]
    print(slope_at_w0)

    C_eq = 0.5 * slope_at_w0

    L_eq = 1 / (w0**2 * C_eq)



    lc = lc_resonator(freq=freq, L=L_eq, C=C_eq)
    port=rf.Circuit.Port(freq, name="P1", z0=500)
    port2=rf.Circuit.Port(freq, name="P2", z0=500)
    cnx2 = [[(port, 0), (lc, 0)],
            [(lc, 1), (port2, 0)]]

    ckt = rf.Circuit(cnx2, name="LOM_resonator")

    ntw2 = ckt.network

    ntw2.plot_s_db()
    ntw.plot_s_db()

    print("C_eq =", C_eq)
    print("L_eq =", L_eq)

    return C_eq, L_eq



#Full LOM fit using Least Squares

def fit_lom(freq, d, Cc1, Cc2, Ctog1, Ctog2, Z0=50,
    n_dense=100, n_kappa=0.75, w0_window_frac=0.0005, n_w0=20, t=0, h=500e-6, has_metal_backside=False, verbose=False):

    cpw_network = cpw_resonator_network(freq, d, Cc1, Cc2, Ctog1, Ctog2, t=t, h=h, has_metal_backside=has_metal_backside)

    f0 = resonance_from_res11(cpw_network)
    k = fwhm_from_res11(cpw_network)

    freq_new = rf.Frequency(f0 - 2*k, f0 + 2*k, 500_001, unit='Hz')

    cpw_network = cpw_resonator_network(
        freq_new,
        d,
        Cc1,
        Cc2,
        Ctog1,
        Ctog2,
        has_metal_backside=has_metal_backside,
        t=t,
        h=h,
    )

    #initial Leff, Ceff guess
    if Cc1 > Cc2:
        lom_network, f0_cpw, f0_lom, k_cpw, k_lom, Ceff, Leff = approximate_LOM_network(freq, cpw_network, Cc1, Cc2, Z0, n_dense, n_kappa, w0_window_frac=w0_window_frac, n_w0=n_w0)
    else:
        lom_network, f0_cpw, f0_lom, k_cpw, k_lom, Ceff, Leff = approximate_LOM_network(freq, cpw_network, Cc2, Cc1, Z0, n_dense, n_kappa, w0_window_frac=w0_window_frac, n_w0=n_w0)

    
    #make residuals for least squares fit
    residuals_win = make_windowed_residuals(freq_new, cpw_network, Cc1, Cc2, f0=f0, width_hz=k_cpw, n_widths=1)

    #least squares fit for Leff, Ceff
    res2 = least_squares(
        residuals_win,
        x0=(Leff, Ceff),
        bounds=((Leff-2e-10, Ceff-2e-13), (Leff+2e-10, Ceff+2e-13)),
        method='trf',
        jac='3-point',
        diff_step=1e-4,
        x_scale='jac',
        xtol=1e-15, ftol=1e-15, gtol=1e-15,
        max_nfev=200,
        verbose=2,
    )
    
    Leff=res2.x[0]
    Ceff=res2.x[1]

    #print output message if true
    if verbose==True:
        print("x =", res2.x)   
        print("Fit status:", res2.status, " success:", res2.success, " cost:", res2.cost)

    #build final lom network

    lom_network = lc_resonator_network(Cc1=Cc1, Cc2=Cc2, Leff=Leff,  Ceff=Ceff, freq=freq_new)


    
    #final f0, kappa measurements

    f0_lom = resonance_from_res11(lom_network)
    f0_cpw = resonance_from_res11(cpw_network)

    k_lom = fwhm_from_res11(lom_network)
    k_dat = fwhm_from_res11(cpw_network)

    #plot_lom_vs_data_re_im(lom_network, cpw_network)

    # freq_loaded = rf.Frequency(f0 - 1e9, f0 + 1e9, 400_001, unit='Hz')

    # #find g
    # lc_loaded_network = lc_resonator_loaded_network(freq_loaded, Leff=Leff, Ceff=Ceff, Cc1=Cc1, Cc2=Cc2, Lload1=Lload1, Cload1=Cload1, Lload2=Lload2, Cload2=Cload2 )
    # cpw_loaded_network = cpw_resonator_loaded_network(freq_loaded, d, Cc1=Cc1, Cc2=Cc2, Ctog1=Ctog1, Ctog2=Ctog2, 
    #     Lload1=Lload1, Cload1=Cload1, Lload2=Lload2, Cload2=Cload2, 
    #     has_metal_backside=has_metal_backside,
    #     t=t,
    #     h=h,
    #     )
    
    # #plot_lom_vs_data_re_im(lc_loaded_network, cpw_loaded_network)

    # #find shifted frequencies
    # lc_shifted_frequencies = resonances_from_res11(lc_loaded_network)
    # cpw_shifted_frequencies = resonances_from_res11(cpw_loaded_network)




    # frame = inspect.currentframe()
    # args, _, _, values = inspect.getargvalues(frame)
    # arg_dict = {arg: values[arg] for arg in args}
    
    error_f0 = abs(f0_cpw - (f0_lom)) / (f0_cpw) * 100
    error_kappa = abs((k_cpw) - (k_lom))/(k_cpw) * 100

    # cpw_shift = np.abs(f0_cpw/1e9 - (np.max(cpw_shifted_frequencies)))
    # lc_shift = np.abs(f0_lom/1e9 - (np.max(lc_shifted_frequencies)))
    # error_shift = abs(cpw_shift - (lc_shift)) / (cpw_shift) * 100
    # cpw_f1 =np.max(cpw_shifted_frequencies)
    # lc_f1 = np.max(lc_shifted_frequencies)
    # error_shifted_frequencies = abs(cpw_f1 - (lc_f1)) / (cpw_f1) * 100

    

    output = {
        # 'CPW Bare Frequency (GHz)': f0,
        'CPW length': d,
        # 'Load 1 Capacitance (F)':Cload1,
        # 'Load 1 Inductance (H)':Lload1,
        # #'Load 1 Resonant Frequency (GHz)': Load1_Res,
        # 'Load 2 Capacitance (F)': Cload2,
        # 'Load 2 Inductance (H)':Lload2,
        #'Load 2 Resonant Frequency (GHz)': Load2_Res,
        'Cc1': Cc1, #C to feedline
        'Cc2': Cc2, 
        'Ctog': Ctog1,
        'Cclaw': Ctog2,
        'LC C': Ceff,
        'LC L': Leff,
        #'f+': f1_cpw,
        'CPW Frequency (GHz)': f0_cpw,
        'LC Frequency (GHz)': f0_lom,
        'Frequency Error (%)': error_f0,
        'CPW Kappa (MHz)': k_cpw * 1000,
        'LC Kappa (MHz)': k_lom * 1000,
        'Linewidth Error (%)': error_kappa,
        # 'CPW Shifted Frequencies (GHz)': cpw_shifted_frequencies,
        # 'LC Shifted Frequencies (GHz)': lc_shifted_frequencies,
        # 'Shifted Frequency Error (%)':error_shifted_frequencies,
        # 'CPW Shift (GHz)': cpw_shift,
        # 'LC Shift (Ghz)': lc_shift,
        # 'Shift Error (%)': error_shift,
        'Cost': res2.cost,
    }
    return output

@dataclass
class LOMFitter:


    # Required user inputs
    freq: Any            # typically skrf.Frequency or array
    d: float
    Cc1: float
    Cc2: float
    Ctog1: float
    Ctog2: float

    # Options 
    Z0: float = 50
    n_dense: int = 100
    n_kappa: float = 0.75
    w0_window_frac: float = 0.0005
    n_w0: int = 20
    t: float = 0
    h: float = 500e-6
    has_metal_backside: bool = False
    verbose: bool = False

    # Internal cached state after fitting
    _fit_done: bool = field(default=False, init=False)
    _Leff: Optional[float] = field(default=None, init=False)
    _Ceff: Optional[float] = field(default=None, init=False)

    _f0_cpw: Optional[float] = field(default=None, init=False)
    _k_cpw: Optional[float] = field(default=None, init=False)
    _f0_lom: Optional[float] = field(default=None, init=False)
    _k_lom: Optional[float] = field(default=None, init=False)
    _cost: Optional[float] = field(default=None, init=False)

    _freq_new: Optional[Any] = field(default=None, init=False)
    _cpw_network: Any = field(default=None, init=False)
    _lom_network: Any = field(default=None, init=False)

    # -------------------------
    # Core internal pipeline
    # -------------------------
    def _run_fit(self) -> None:
        """Runs the full least-squares fit once and caches results."""
        if self._fit_done:
            return

        # Build CPW network on provided frequency to get initial f0,k
        cpw_network = cpw_resonator_network(
            self.freq,
            self.d,
            self.Cc1,
            self.Cc2,
            self.Ctog1,
            self.Ctog2,
            t=self.t,
            h=self.h,
            has_metal_backside=self.has_metal_backside,
        )
        f0 = resonance_from_res11(cpw_network)
        k = fwhm_from_res11(cpw_network)

        # Re-simulate densely around resonance
        freq_new = rf.Frequency(f0 - 2 * k, f0 + 2 * k, 500_001, unit="Hz")
        cpw_network = cpw_resonator_network(
            freq_new,
            self.d,
            self.Cc1,
            self.Cc2,
            self.Ctog1,
            self.Ctog2,
            t=self.t,
            h=self.h,
            has_metal_backside=self.has_metal_backside,
        )

        if self.Cc1 > self.Cc2:
            _, f0_cpw, f0_lom, k_cpw, k_lom, Ceff0, Leff0 = approximate_LOM_network(
                self.freq,
                cpw_network,
                self.Cc1,
                self.Cc2,
                self.Z0,
                self.n_dense,
                self.n_kappa,
                w0_window_frac=self.w0_window_frac,
                n_w0=self.n_w0,
            )
        else:
            _, f0_cpw, f0_lom, k_cpw, k_lom, Ceff0, Leff0 = approximate_LOM_network(
                self.freq,
                cpw_network,
                self.Cc2,
                self.Cc1,
                self.Z0,
                self.n_dense,
                self.n_kappa,
                w0_window_frac=self.w0_window_frac,
                n_w0=self.n_w0,
            )

        # Windowed residuals for least squares
        residuals_win = make_windowed_residuals(
            freq_new,
            cpw_network,
            self.Cc1,
            self.Cc2,
            f0=f0,
            width_hz=k_cpw,
            n_widths=1,
        )

        # Fit
        res2 = least_squares(
            residuals_win,
            x0=(Leff0, Ceff0),
            bounds=((Leff0 - 2e-10, Ceff0 - 2e-13), (Leff0 + 2e-10, Ceff0 + 2e-13)),
            method="trf",
            jac="3-point",
            diff_step=1e-4,
            x_scale="jac",
            xtol=1e-15,
            ftol=1e-15,
            gtol=1e-15,
            max_nfev=200,
            verbose=2 if self.verbose else 0,
        )

        Leff = float(res2.x[0])
        Ceff = float(res2.x[1])

        if self.verbose:
            print("x =", res2.x)
            print("Fit status:", res2.status, " success:", res2.success, " cost:", res2.cost)

        # Final LOM network and metrics
        lom_network = lc_resonator_network(Cc1=self.Cc1, Cc2=self.Cc2, Leff=Leff, Ceff=Ceff, freq=freq_new)

        f0_lom_final = resonance_from_res11(lom_network)
        f0_cpw_final = resonance_from_res11(cpw_network)

        k_lom_final = fwhm_from_res11(lom_network)
        k_cpw_final = fwhm_from_res11(cpw_network)

        # Cache
        self._fit_done = True
        self._Leff, self._Ceff = Leff, Ceff
        self._f0_cpw, self._k_cpw = f0_cpw_final, k_cpw_final
        self._f0_lom, self._k_lom = f0_lom_final, k_lom_final
        self._cost = float(res2.cost)

        self._freq_new = freq_new
        self._cpw_network = cpw_network
        self._lom_network = lom_network

    # -------------------------
    # Public API requested
    # -------------------------
    def fit_leff_ceff(self) -> Tuple[float, float]:
        """
        Option 1: just get Leff and Ceff (runs fit if needed).
        """
        self._run_fit()
        return self._Leff, self._Ceff  # type: ignore[return-value]

    def f0_kappa_error(self) -> Tuple[float, float]:
        """
        Option 2: get % error for f0 and kappa (runs fit if needed).
        Returns:
          (error_f0_percent, error_kappa_percent)
        """
        self._run_fit()
        assert self._f0_cpw is not None and self._f0_lom is not None
        assert self._k_cpw is not None and self._k_lom is not None

        error_f0 = abs(self._f0_cpw - self._f0_lom) / self._f0_cpw * 100.0
        error_kappa = abs(self._k_cpw - self._k_lom) / self._k_cpw * 100.0
        return float(error_f0), float(error_kappa)

    def frequency_shift(
        self,
        loads: Dict[str, float],
        span_hz: float = 1e9,
        npoints: int = 400_001,
    ) -> Dict[str, Any]:
        """
        Option 3: compute shifted resonances given some loads.

        `loads` keys (suggested):
          Lload1, Cload1, Lload2, Cload2

        Returns a dict containing:
          - f0_cpw_bare, f0_lom_bare
          - cpw_shifted_frequencies, lom_shifted_frequencies
          - cpw_shift, lom_shift   (as max resonance shift relative to bare)
        """
        self._run_fit()
        assert self._Leff is not None and self._Ceff is not None
        assert self._f0_cpw is not None and self._f0_lom is not None

        # Frequency sweep for loaded case
        freq_loaded = rf.Frequency(self._f0_cpw - span_hz, self._f0_cpw + span_hz, npoints, unit="Hz")

        Lload1 = loads.get("Lload1", 0.0)
        Cload1 = loads.get("Cload1", 0.0)
        Lload2 = loads.get("Lload2", 0.0)
        Cload2 = loads.get("Cload2", 0.0)

        lc_loaded_network = lc_resonator_loaded_network(
            freq_loaded,
            Leff=self._Leff,
            Ceff=self._Ceff,
            Cc1=self.Cc1,
            Cc2=self.Cc2,
            Lload1=Lload1,
            Cload1=Cload1,
            Lload2=Lload2,
            Cload2=Cload2,
        )

        cpw_loaded_network = cpw_resonator_loaded_network(
            freq_loaded,
            self.d,
            Cc1=self.Cc1,
            Cc2=self.Cc2,
            Ctog1=self.Ctog1,
            Ctog2=self.Ctog2,
            Lload1=Lload1,
            Cload1=Cload1,
            Lload2=Lload2,
            Cload2=Cload2,
            has_metal_backside=self.has_metal_backside,
            t=self.t,
            h=self.h,
        )

        lom_shifted_frequencies = resonances_from_res11(lc_loaded_network)
        cpw_shifted_frequencies = resonances_from_res11(cpw_loaded_network)

        # compare bare f0 to the maximum shifted resonance (if multiple).
        cpw_f1 = float(np.max(cpw_shifted_frequencies))
        lom_f1 = float(np.max(lom_shifted_frequencies))

        cpw_shift = abs(self._f0_cpw - cpw_f1)
        lom_shift = abs(self._f0_lom - lom_f1)

        return {
            "f0_cpw_bare_hz": float(self._f0_cpw),
            "f0_lom_bare_hz": float(self._f0_lom),
            "cpw_shifted_frequencies_hz": cpw_shifted_frequencies,
            "lom_shifted_frequencies_hz": lom_shifted_frequencies,
            "cpw_shift_hz": float(cpw_shift),
            "lom_shift_hz": float(lom_shift),
            "cpw_f1_hz": cpw_f1,
            "lom_f1_hz": lom_f1,
        }

    def summary(self) -> Dict[str, Any]:
        self._run_fit()
        err_f0, err_k = self.f0_kappa_error()

        return {
            "CPW length": self.d,
            "Cc1": self.Cc1,
            "Cc2": self.Cc2,
            "Ctog": self.Ctog1,
            "Cclaw": self.Ctog2,
            "LC C": self._Ceff,
            "LC L": self._Leff,
            "CPW Frequency (GHz)": self._f0_cpw,
            "LC Frequency (GHz)": self._f0_lom,
            "Frequency Error (%)": err_f0,
            "CPW Kappa (MHz)": self._k_cpw * 1000 if self._k_cpw is not None else None,
            "LC Kappa (MHz)": self._k_lom * 1000 if self._k_lom is not None else None,
            "Linewidth Error (%)": err_k,
            "Cost": self._cost,
        }
    
    def plot(self) -> None:
        """Optional: plot S-parameters of CPW and LOM networks."""
        self._run_fit()
        if self._cpw_network is not None and self._lom_network is not None:
            plot_lom_vs_data_re_im(self._lom_network, self._cpw_network)


# -------------------------
# Example usage
# -------------------------
# fitter = LOMFitter(freq=freq, d=d, Cc1=Cc1, Cc2=Cc2, Ctog1=Ctog1, Ctog2=Ctog2, verbose=True)
# Leff, Ceff = fitter.fit_leff_ceff()
# err_f0, err_k = fitter.f0_kappa_error()
# shifts = fitter.frequency_shift({"Lload1": 1e-9, "Cload1": 1e-15, "Lload2": 0.0, "Cload2": 0.0})
# out = fitter.summary()


# @dataclass
# class FosterSynthesis:
#     """
#     Foster synthesis wrapper for extracting an equivalent (C_eq, L_eq)
#     from a CPW resonator network.

#     Core public methods:
#       - extract_ceq_leq(): returns (C_eq, L_eq) and caches results
#       - resonance_frequencies(): returns detected f0 candidates (Hz)
#       - build_lom_network(): returns the LC (LOM) network built from extracted params
#     """

#     # Required
#     freq: Any

#     # CPW + geometry/material parameters 
#     w: float = 11.7e-6
#     s: float = 5.1e-6
#     t: float = 0
#     h: float = 500e-6
#     rho: float = 1e-19
#     ep_r: float = 11.45
#     has_metal_backside: bool = True
#     tand: float = 0
#     d: float = 0.007

#     # Port settings
#     z0_port: float = 500.0

#     # Peak picking controls (optional but handy)
#     peak_index: int = 0  # which detected peak to use (0 = first)
#     plot: bool = False   # optionally plot s_db
#     verbose: bool = True

#     # Cached internals
#     _cpw_network: Any = field(default=None, init=False)
#     _lom_network: Any = field(default=None, init=False)
#     _f0s: Optional[np.ndarray] = field(default=None, init=False)
#     _w0: Optional[float] = field(default=None, init=False)
#     _slope_at_w0: Optional[float] = field(default=None, init=False)
#     _C_eq: Optional[float] = field(default=None, init=False)
#     _L_eq: Optional[float] = field(default=None, init=False)

#     # -------------------------
#     # Internal builders
#     # -------------------------
#     def _build_cpw_network(self) -> Any:
#         """Construct and cache the CPW resonator 2-port network."""
#         if self._cpw_network is not None:
#             return self._cpw_network

#         cpw = CPW(
#             self.freq,
#             w=self.w,
#             s=self.s,
#             t=self.t,
#             h=self.h,
#             rho=self.rho,
#             ep_r=self.ep_r,
#             has_metal_backside=self.has_metal_backside,
#             tand=self.tand,
#         )

#         line = cpw.line(d=self.d, unit="m", name="cpw line")
#         port1 = rf.Circuit.Port(self.freq, name="P1", z0=self.z0_port)
#         port2 = rf.Circuit.Port(self.freq, name="P2", z0=self.z0_port)

#         cnx = [
#             [(port1, 0), (line, 0)],
#             [(line, 1), (port2, 0)],
#         ]

#         ckt = rf.Circuit(cnx, name="Port_Resonator")
#         ntw = ckt.network

#         if self.plot:
#             ntw.plot_s_db()

#         self._cpw_network = ntw
#         return ntw

#     def _detect_resonances(self) -> np.ndarray:
#         """
#         Cached in self._f0s.
#         """
#         if self._f0s is not None:
#             return self._f0s

#         ntw = self._build_cpw_network()
#         f = ntw.frequency.f
#         S22 = ntw.s[:, 1, 1]  # complex array

#         # We'll peak-pick on |S22| by default (closest intent, robust).
#         mag = np.abs(S22)

#         peaks, _ = find_peaks(mag)
#         f0s = f[peaks]

#         self._f0s = f0s

#         if self.verbose:
#             print("Detected resonance candidates (Hz):", f0s)

#         return f0s

#     # -------------------------
#     # Public API
#     # -------------------------
#     def resonance_frequencies(self) -> np.ndarray:
#         """Return detected resonance frequency candidates (Hz)."""
#         return self._detect_resonances()

#     def extract_ceq_leq(self) -> Tuple[float, float]:
#         """
#         Perform Foster extraction:
#           - compute Yin = Y11 - (Y12*Y21)/(Y22+YL)
#           - B = Im(Yin)
#           - slope dB/dw at chosen resonance w0
#           - C_eq = 0.5 * slope
#           - L_eq = 1 / (w0^2 * C_eq)

#         Returns (C_eq, L_eq) and caches them.
#         """
#         if self._C_eq is not None and self._L_eq is not None:
#             return self._C_eq, self._L_eq

#         ntw = self._build_cpw_network()
#         f0s = self._detect_resonances()
#         if f0s.size == 0:
#             raise ValueError("No resonance peaks found. Try adjusting peak detection or frequency span.")

#         # pick which resonance to use
#         idx = int(np.clip(self.peak_index, 0, f0s.size - 1))
#         f0 = float(f0s[idx])

#         Y = ntw.y
#         Y11, Y12 = Y[:, 0, 0], Y[:, 0, 1]
#         Y21, Y22 = Y[:, 1, 0], Y[:, 1, 1]

#         YL = 1.0 / self.z0_port
#         Yin = Y11 - (Y12 * Y21) / (Y22 + YL)
#         B = np.imag(Yin)

#         w = 2.0 * np.pi * ntw.frequency.f
#         w0 = 2.0 * np.pi * f0

#         dB_dw = np.gradient(B, w)
#         i0 = int(np.argmin(np.abs(w - w0)))
#         slope_at_w0 = float(dB_dw[i0])

#         C_eq = 0.5 * slope_at_w0
#         if C_eq == 0:
#             raise ZeroDivisionError("C_eq computed as 0; check peak selection and admittance slope.")

#         L_eq = 1.0 / (w0**2 * C_eq)

#         self._w0 = w0
#         self._slope_at_w0 = slope_at_w0
#         self._C_eq = float(C_eq)
#         self._L_eq = float(L_eq)

#         if self.verbose:
#             print("Using f0 (Hz):", f0)
#             print("slope dB/dw at w0:", slope_at_w0)
#             print("C_eq =", C_eq)
#             print("L_eq =", L_eq)

#         return self._C_eq, self._L_eq

#     def build_lom_network(self) -> Any:
#         """
#         Build and cache the equivalent LC (LOM) network from extracted params.
#         Returns the scikit-rf Network.
#         """
#         if self._lom_network is not None:
#             return self._lom_network

#         C_eq, L_eq = self.extract_ceq_leq()

#         lc = lc_resonator(freq=self.freq, L=L_eq, C=C_eq)
#         port1 = rf.Circuit.Port(self.freq, name="P1", z0=self.z0_port)
#         port2 = rf.Circuit.Port(self.freq, name="P2", z0=self.z0_port)

#         cnx = [
#             [(port1, 0), (lc, 0)],
#             [(lc, 1), (port2, 0)],
#         ]

#         ckt = rf.Circuit(cnx, name="LOM_resonator")
#         ntw2 = ckt.network

#         if self.plot:
#             ntw2.plot_s_db()
#             # also plot cpw again for comparison
#             self._build_cpw_network().plot_s_db()

#         self._lom_network = ntw2
#         return ntw2

#     def summary(self) -> Dict[str, float]:
#         C_eq, L_eq = self.extract_ceq_leq()
#         f0s = self.resonance_frequencies()
#         return {
#             "C_eq_F": float(C_eq),
#             "L_eq_H": float(L_eq),
#             "num_resonances_found": float(len(f0s)),
#             "chosen_peak_index": float(self.peak_index),
#             "w0_rad_s": float(self._w0) if self._w0 is not None else float("nan"),
#             "slope_dB_dw": float(self._slope_at_w0) if self._slope_at_w0 is not None else float("nan"),
#         }





def approximate_Ceff_Leff_calculation(Cc1, Cc2, kappa, f_res, Z0=50):
    """
    Args:
        Cc1 (F): LHS coupling capacitor
        Cc2 (F): RHS coupling capacitor
        kappa (Hz): linewidth to match
        f_res (Hz): frequency to match
        Z0 (Ohms): Input impedance in Ohms. Defaults to 50.
    Returns:
        C_approx (F): Capacitance of the approximate matched LC cirucit
        L_approx (H): Inductance of the approximate matched LC cirucit

    """
    k = (kappa) * 2 * np.pi #ANGULAR UNITS !! 
    w0 = 2*np.pi*f_res
    C_approx = (Z0*w0**2*(Cc1)**2)/k- (Cc1 + Cc2) 
    L_approx = 1/(w0**2 * (C_approx + Cc1 + Cc2)) 

    return C_approx, L_approx



# from dataclasses import dataclass
# from typing import Optional, Tuple

# import numpy as np
# import skrf as rf


@dataclass(frozen=True)
class FosterSynthesis:
    """
    Foster-style local synthesis of an effective parallel LC seen at a port:
      Ceff = 0.5 * d(Im{Yin})/dω |_(ω0)
      Leff = 1 / (ω0^2 * Ceff)

    where Yin is the input admittance looking into the 2-port terminated by YL on the
    *other* port

    Notes:
    - This assumes the resonance is well-approximated locally by a simple reactive slope.
    - Units: Hz, rad/s, Farads, Henries.
    """

    YL: float = 1 / 500  # termination admittance on the opposite port
    port: int = 1        # which port we're looking into: 1 or 2
    fit_half_window: int = 5  # number of points on each side for local linear polyfit

    def yin(self, ntw: rf.Network) -> np.ndarray:
        """
          Yin = Y11 - (Y12*Y21)/(Y22 + YL)
        """
        Y = ntw.y

        if self.port == 1:
            Y11 = Y[:, 0, 0]
            Y12 = Y[:, 0, 1]
            Y21 = Y[:, 1, 0]
            Y22 = Y[:, 1, 1]
        elif self.port == 2:
            # looking into port 2, terminate port 1 by YL
            Y11 = Y[:, 1, 1]
            Y12 = Y[:, 1, 0]
            Y21 = Y[:, 0, 1]
            Y22 = Y[:, 0, 0]
        else:
            raise ValueError("port must be 1 or 2")

        return Y11 - (Y12 * Y21) / (Y22 + self.YL)

    def _local_slope_dB_dw(self, w: np.ndarray, B: np.ndarray, w0: float) -> float:
        """Local linear fit slope of B(ω)=Im{Yin} around ω0."""
        i0 = int(np.argmin(np.abs(w - w0)))
        N = int(self.fit_half_window)

        i_lo = max(i0 - N, 0)
        i_hi = min(i0 + N + 1, len(w))

        w_local = w[i_lo:i_hi]
        B_local = B[i_lo:i_hi]

        if len(w_local) < 2:
            raise ValueError("Not enough points for local slope fit. Increase sweep points/window.")

        # slope from linear fit
        slope = np.polyfit(w_local, B_local, 1)[0]
        return float(slope)

    def synthesize_at_frequency(
        self,
        ntw: rf.Network,
        f0_hz: float,
    ) -> Tuple[float, float]:
        """
        Compute (Ceff, Leff) at a specified resonance frequency f0_hz.
        """
        f = ntw.frequency.f
        w = 2 * np.pi * f
        w0 = 2 * np.pi * float(f0_hz)

        Yin = self.yin(ntw)
        B = np.imag(Yin)

        slope_at_w0 = self._local_slope_dB_dw(w, B, w0)

        Ceff = 0.5 * slope_at_w0
        Leff = 1.0 / (w0**2 * Ceff)

        return float(Ceff), float(Leff)

    def synthesize_first_mode_from_s22(
        self,
        ntw: rf.Network,
        m: int = 1,
        n: int = 1,
        prefer_minimum: bool = True,
    ) -> Tuple[float, float, float]:
        """
        Convenience helper: pick the first resonance from S(m,n) and synthesize there.

        Returns (f0_hz, Ceff, Leff).

        - If prefer_minimum=True, finds the first *dip* in |S| (typical for resonator notches).
        - If prefer_minimum=False, finds the first *peak* in |S|.
        """
        # S-parameter indices: m,n are 1-based for user convenience
        s = ntw.s[:, m - 1, n - 1]
        mag = np.abs(s)

        # crude resonance pick without scipy: find first local extremum
        if prefer_minimum:
            y = -mag  # maxima of -|S| correspond to minima of |S|
        else:
            y = mag

        # local maxima detection (simple, no dependencies)
        dy1 = y[1:-1] - y[:-2]
        dy2 = y[1:-1] - y[2:]
        peaks = np.where((dy1 > 0) & (dy2 > 0))[0] + 1

        if len(peaks) == 0:
            raise ValueError("No resonance-like extremum found in |S|. Provide f0_hz explicitly.")

        f0_hz = float(ntw.frequency.f[peaks[0]])
        Ceff, Leff = self.synthesize_at_frequency(ntw, f0_hz)
        return f0_hz, Ceff, Leff

def plot_two_networks_s_db(
    ntw_main: rf.Network,
    ntw_equiv: rf.Network,
    *,
    labels: tuple[str, str] = ("Main", "Equivalent"),
    ports: Sequence[tuple[int, int]] | None = None,
    x_unit: str = "ghz",
    title: str | None = None,
):
    """
    Overlay S-parameter dB traces for two scikit-rf Networks on the same axes.

    - labels: legend names for (ntw_main, ntw_equiv)
    - ports: list of (m,n) 1-based S-params to plot; default plots all for the network's nports
    - x_unit: "hz" | "khz" | "mhz" | "ghz"

    Example:
        plot_two_networks_s_db(ntw, ntw_lom, labels=("CPW", "Foster LOM"), ports=[(2,2)])
    """
    import matplotlib.pyplot as plt

    def _xscale(f_hz: np.ndarray, unit: str) -> tuple[np.ndarray, str]:
        unit = unit.lower()
        scale = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9}.get(unit)
        if scale is None:
            raise ValueError("x_unit must be one of: hz, khz, mhz, ghz")
        return f_hz / scale, unit.upper()

    # sanity: same frequency grid
    f1 = ntw_main.frequency.f
    f2 = ntw_equiv.frequency.f
    if len(f1) != len(f2) or not np.allclose(f1, f2):
        raise ValueError("Networks must share the same frequency grid to plot together.")

    nports = ntw_main.nports
    if ntw_equiv.nports != nports:
        raise ValueError("Networks must have the same number of ports to plot together.")

    if ports is None:
        ports = [(m, n) for m in range(1, nports + 1) for n in range(1, nports + 1)]

    x, xlab = _xscale(f1, x_unit)

    plt.figure()
    for (m, n) in ports:
        s_main = ntw_main.s[:, m - 1, n - 1]
        s_equiv = ntw_equiv.s[:, m - 1, n - 1]

        y_main = 20 * np.log10(np.abs(s_main))
        y_equiv = 20 * np.log10(np.abs(s_equiv))

        # label once per trace pair
        plt.plot(x, y_main, label=f"{labels[0]} S{m}{n}")
        plt.plot(x, y_equiv, linestyle="--", label=f"{labels[1]} S{m}{n}")

    plt.xlabel(f"Frequency ({xlab})")
    plt.ylabel("S (dB)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.title(title or "Overlay S-parameters (dB)")
    plt.show()

    def _fs_plot_networks(self, ntw_main: rf.Network, ntw_equiv: rf.Network, **kwargs):
        return plot_two_networks_s_db(ntw_main, ntw_equiv, **kwargs)

    FosterSynthesis.plot_networks = _fs_plot_networks