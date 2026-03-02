
import numpy as np
import skrf as rf
from skrf.media import CPW
from scipy.optimize import minimize
from scipy.signal import find_peaks

def coupling_capacitor(C, freq, name='cc'):
            return rf.Circuit.SeriesImpedance(frequency=freq, name=name, z0=50,
                                        Z=1/(1j*freq.w*C))

def shunt_capacitor(C, freq, name='ctog'):
        return rf.Circuit.ShuntAdmittance(frequency=freq, name=name, z0=50,
                                    Y=1j*freq.w*C)

def lc_resonator(L, C, freq, z0=50, name='lc'):
        return rf.Circuit.ShuntAdmittance(frequency=freq, name=name, z0=z0,
                                    Y=1j*freq.w*C + 1/(1j*freq.w*L))


def lc_resonator_network_2port(Leff, Ceff, Cc1, Cc2, freq):

        cc1 = coupling_capacitor(C=Cc1,freq=freq, name='cc1')
        lc=lc_resonator(L=Leff, C=Ceff, freq=freq, name='lc')
        cc2 = coupling_capacitor(C=Cc2,freq=freq, name='cc2')

        open = rf.Circuit.Open(freq, name='open')
        port1=rf.Circuit.Port(freq, name="P1", z0=50)
        port2=rf.Circuit.Port(freq, name="P2", z0=50)

        cnx2 = [[(port1, 0), (cc1, 0)],
        [(cc1, 1), (lc, 0)],
            [(lc, 1), (cc2, 0)],
            [(cc2, 1), (port2, 0)]]
        
        ckt = rf.Circuit(cnx2, name="Port_Resonator")
        network_port = ckt.network
        return network_port



def lc_resonator_network(Leff, Ceff, Cc1, Cc2, freq):

    cc1 = coupling_capacitor(C=Cc1,freq=freq, name='cc1')
    lc=lc_resonator(L=Leff, C=Ceff, freq=freq, name='lc')
    cc2 = shunt_capacitor(C=Cc2,freq=freq, name='cc2')

    open = rf.Circuit.Open(freq, name='open')
    port50=rf.Circuit.Port(freq, name="P1", z0=50)

    cnx2 = [[(port50, 0), (cc1, 0)],
       [(cc1, 1), (lc, 0)],
         [(lc, 1), (cc2, 0)],
        [(cc2, 1), (open, 0)]]
    
    ckt = rf.Circuit(cnx2, name="Port_Resonator")
    network_port = ckt.network
    return network_port


def lc_resonator_loaded_network(freq, Leff, Ceff, Cc1, Cc2, Lload1, Cload1, Lload2, Cload2):

    cc1 = coupling_capacitor(C=Cc1,freq=freq, name='cc1')
    lc=lc_resonator(L=Leff, C=Ceff, freq=freq, name='lc')
    cc2 = coupling_capacitor(C=Cc2,freq=freq, name='cc2')
    cc_port = coupling_capacitor(C=1e-16,freq=freq, name='cc_port')

    #loads
    load1 =lc_resonator(L=Lload1, C=Cload1, freq=freq, name='load1')
    load2 =lc_resonator(L=Lload2, C=Cload2, freq=freq, name='load2')

    open = rf.Circuit.Open(freq, name='open')
    port50=rf.Circuit.Port(freq, name="P1", z0=50)

    cnx2 = [[(port50, 0), (cc_port, 0)],
       [(cc_port, 1), (load1, 0)],
        [(load1, 1), (cc1, 0)],
       [(cc1, 1), (lc, 0)],
         [(lc, 1), (cc2, 0)],
        [(cc2, 1), (load2, 0)],
        [(load2, 1), (open, 0)]]
    
    ckt = rf.Circuit(cnx2, name="Port_Resonator")
    network_port = ckt.network
    return network_port

def lc_resonator_network_withCtog(Leff, Ceff, Cc1, Cc2, Ctog1, Ctog2, freq):
    cc1 = coupling_capacitor(C=Cc1,freq=freq, name='cc1')
    lc=lc_resonator(L=Leff, C=Ceff, freq=freq, name='lc')
    cc2 = shunt_capacitor(C=Cc2,freq=freq, name='cc2')

    ctog1 = shunt_capacitor(C=Ctog1, freq=freq, name='ctog1')
    ctog2 = shunt_capacitor(C=Ctog2,freq=freq, name='ctog2')

    open = rf.Circuit.Open(name='open', frequency=freq)
    port50=rf.Circuit.Port(freq, name="P1", z0=50)

    cnx2 = [[(port50, 0), (cc1, 0)],
       [(cc1, 1), (ctog1, 0)],
         [(ctog1, 1), (lc, 0)],
        [(lc, 1),  (ctog2, 0)],
        [(ctog2, 1), (cc2, 0)],
        [(cc2, 1), (open, 0)]]
    
    ckt = rf.Circuit(cnx2, name="Port_Resonator")
    network_port = ckt.network
    return network_port


def cpw_resonator_network_2port(freq, d, Cc1, Cc2, Ctog1, Ctog2,  w = 11.7e-6,   
                s = 5.1e-6, 
                t=0, 
                h=500e-6, 
                rho=1e-19,
                ep_r = 11.45,
                has_metal_backside=True,
                tand=0,):
        # CPW definition
    cpw = CPW(freq, w=w ,   
                s=s,
                t=t, 
                h=h, 
                rho=rho, 
                ep_r=ep_r,
                has_metal_backside=has_metal_backside,
                tand=tand,
                )


    #cc1 = cpw.capacitor(C=Cc1, name='cc1')  # coupling capacitor at input
    # CPW line
    line = cpw.line(d=d, unit='m', name='cpw_line')

    cc1 = coupling_capacitor(C=Cc1,freq=freq, name='cc1')
    # lc=lc_resonator(L=Leff, C=Ceff, freq=freq, name='lc')
    cc2 = coupling_capacitor(C=Cc2,freq=freq, name='cc2')
    #cc_port = coupling_capacitor(C=1e-16,freq=freq, name='cc_port')

    ctog1 = cpw.shunt_capacitor(C=Ctog1, freq=freq, name='ctog1')
    ctog2 = cpw.shunt_capacitor(C=Ctog2, freq=freq, name='ctog2')

    open = rf.Circuit.Open(freq, name='open')
    port1=rf.Circuit.Port(freq, name="P1", z0=50)
    port2=rf.Circuit.Port(freq, name="P2", z0=50)


    cnx2 = [[(port1, 0), (cc1, 0)],
        [(cc1, 1), (ctog1, 0)],
        [(ctog1, 1), (line, 0)],
        [(line, 1), (ctog2, 0)],
        [(ctog2, 1), (cc2, 0)],
        [(cc2, 1), (port2, 0)]]
    
    ckt = rf.Circuit(cnx2, name="Port_Resonator")
    network_port = ckt.network
    return network_port


def cpw_resonator_network(freq, d, Cc1, Cc2, Ctog1, Ctog2,  w = 11.7e-6,   
                s = 5.1e-6, #spacing = 5.1um
                t=0, #thickness = 200nm
                h=500e-6, #substrate height = 525um
                rho=1e-19, #closest to 0 this thing goes
                ep_r = 11.45, #ultracold silicon
                has_metal_backside=True,
                tand=0,):
        # CPW definition
    cpw = CPW(freq, w=w ,   
                s=s, #spacing = 5.1um
                t=t, #thickness = 200nm
                h=h, #substrate height = 525um
                rho=rho, #closest to 0 this thing goes
                ep_r=ep_r, #ultracold silicon
                has_metal_backside=has_metal_backside,
                tand=tand,
                )


    #cc1 = cpw.capacitor(C=Cc1, name='cc1')  # coupling capacitor at input
    # CPW line
    line = cpw.line(d=d, unit='m', name='cpw_line')

    cc1 = coupling_capacitor(C=Cc1,freq=freq, name='cc1')
    # lc=lc_resonator(L=Leff, C=Ceff, freq=freq, name='lc')
    cc2 = shunt_capacitor(C=Cc2,freq=freq, name='cc2')
    #cc_port = coupling_capacitor(C=1e-16,freq=freq, name='cc_port')

    ctog1 = cpw.shunt_capacitor(C=Ctog1, freq=freq, name='ctog1')
    ctog2 = cpw.shunt_capacitor(C=Ctog2, freq=freq, name='ctog2')

    open = rf.Circuit.Open(freq, name='open')
    port50=rf.Circuit.Port(freq, name="P1", z0=50)


    cnx2 = [[(port50, 0), (cc1, 0)],
        [(cc1, 1), (ctog1, 0)],
        [(ctog1, 1), (line, 0)],
        [(line, 1), (ctog2, 0)],
        [(ctog2, 1), (cc2, 0)],
        [(cc2, 1), (open, 0)]]
    
    ckt = rf.Circuit(cnx2, name="Port_Resonator")
    network_port = ckt.network
    return network_port


def resonance_from_res11(ntwk):
    """Estimate resonance frequency from S11 data."""
    f = ntwk.frequency.f
    ReS11 = np.real(ntwk.s[:, 0, 0])
    min_index = np.argmin(ReS11)
    f0 = f[min_index]
    return f0

def fwhm_from_res11(ntwk):
    #half maximum
    res11 = np.real(ntwk.s[:, 0, 0])
    half_max = (np.max(res11) + np.min(res11)) / 2
    indices = np.where(np.diff(np.sign(res11 - half_max)))[0]
    if len(indices) < 2:
        raise ValueError("Could not determine FWHM from Re(S11) data")  # FWHM not found
    f1 = ntwk.frequency.f[indices[0]]
    f2 = ntwk.frequency.f[indices[1]]
    fwhm = f2 - f1
    return fwhm

def make_windowed_residuals(freq, data_ntw, Cc1, Cc2, f0, width_hz, n_widths=1, eps=1e-18):
    data_aligned = align_data_network_to_freq(data_ntw, freq)
    f = freq.f
    mask = (f > f0 - n_widths*width_hz) & (f < f0 + n_widths*width_hz)

    freq_win = rf.Frequency.from_f(f[mask], unit='Hz')
    s_data = data_aligned.s[mask, 0, 0]

    # "AC scale" of the data: remove mean baseline, then RMS
    s0 = np.mean(s_data)
    S_scale = np.sqrt(np.mean(np.abs(s_data - s0)**2)) + eps

    def residuals(x):
        Leff, Ceff = float(x[0]), float(x[1])
        lom = lc_resonator_network(Leff=Leff, Ceff=Ceff, Cc1=Cc1, Cc2=Cc2, freq=freq_win)
        r = lom.s[:, 0, 0] - s_data
        # dimensionless, comparable across different resonance depths/baselines
        return np.concatenate([r.real, r.imag]) / S_scale

    return residuals


def _interp_complex(x_new, x_old, y_old_complex):
    """
    Simple linear interpolation of complex-valued data by interpolating
    Re and Im separately.
    """
    yr = np.interp(x_new, x_old, np.real(y_old_complex))
    yi = np.interp(x_new, x_old, np.imag(y_old_complex))
    return yr + 1j*yi



def align_data_network_to_freq(data_ntw, freq):
    """
    Returns a 1-port Network with the same frequency grid as `freq`.
    If already aligned, returns `data_ntw` unchanged.
    """
    f_old = data_ntw.frequency.f
    f_new = freq.f

    if len(f_old) == len(f_new) and np.allclose(f_old, f_new, rtol=0, atol=0):
        return data_ntw

    s11_old = data_ntw.s[:, 0, 0]
    s11_new = _interp_complex(f_new, f_old, s11_old)

    # Build a 1-port network on the new grid
    s_new = s11_new.reshape(-1, 1, 1)
    return rf.Network(frequency=freq, s=s_new, z0=data_ntw.z0)


def build_sparse_data_points(data_ntw, n_dense=0, n_kappa=2.0):
        """
        Build a sparse, physically-motivated set of data points:
        - max Im(S11)
        - min Im(S11)
        - min Re(S11) (resonance)
        Optionally add dense samples around resonance.

        Returns:
            data_points : list of (omega, ReS11, ImS11)
        """

        f_hz = data_ntw.frequency.f              # Hz
        w    = 2 * np.pi * f_hz

        S11  = data_ntw.s[:, 0, 0]
        ReS11 = np.real(S11)
        ImS11 = np.imag(S11)

        data_points = []

        
        # # 1. max Im(S11)
        # idx_im_max = np.argmax(ImS11)
        # data_points.append(
        #     (w[idx_im_max], ReS11[idx_im_max], ImS11[idx_im_max])
        # )

        # # 2. min Im(S11)
        # idx_im_min = np.argmin(ImS11)
        # data_points.append(
        #     (w[idx_im_min], ReS11[idx_im_min], ImS11[idx_im_min])
        # )

        # 3. min Re(S11) → resonance
        idx_re_min = np.argmin(ReS11)
        data_points.append(
            (w[idx_re_min], ReS11[idx_re_min], ImS11[idx_re_min])
        )

        # Optional: dense window around resonance
        if n_dense > 0:
            f0 = f_hz[idx_re_min]
            kappa = fwhm_from_res11(data_ntw)  # Hz


            f_dense = np.linspace(
                f0 - n_kappa * kappa,
                #f0,
                f0 + n_kappa/2 * kappa,
                n_dense + 2
            )[1:-1]

            # interpolate complex S11
            Re_dense = np.interp(f_dense, f_hz, ReS11)
            Im_dense = np.interp(f_dense, f_hz, ImS11)

            for f, r, i in zip(f_dense, Re_dense, Im_dense):
                data_points.append((2*np.pi*f, r, i))

        return data_points



def cpw_resonator_loaded_network(freq, d, Cc1, Cc2, Ctog1, Ctog2, Lload1, Cload1, Lload2, Cload2, 
                w = 11.7e-6,   
                s = 5.1e-6, #spacing = 5.1um
                t=0, #thickness = 200nm
                h=500e-6, #substrate height = 525um
                rho=1e-19, #closest to 0 this thing goes
                ep_r = 11.45, #ultracold silicon
                has_metal_backside=True,
                tand=0,
                ):
        # CPW definition
    cpw = CPW(freq, w = w,   
                s = s, #spacing = 5.1um
                t=t, #thickness = 200nm
                h=h, #substrate height = 525um
                rho=rho, #closest to 0 this thing goes
                ep_r = ep_r, #ultracold silicon
                has_metal_backside=has_metal_backside,
                tand=tand,
                )

    #cc1 = cpw.capacitor(C=Cc1, name='cc1')  # coupling capacitor at input
    # CPW line
    line = cpw.line(d=d, unit='m', name='cpw_line')

    cc1 = coupling_capacitor(C=Cc1,freq=freq, name='cc1')
    cc2 = coupling_capacitor(C=Cc2,freq=freq, name='cc2')

    cc_port = coupling_capacitor(C=1e-16,freq=freq, name='cc_port')

    ctog1 = cpw.shunt_capacitor(C=Ctog1, freq=freq, name='ctog1')
    ctog2 = cpw.shunt_capacitor(C=Ctog2, freq=freq, name='ctog2')

    #loads
    load1 =lc_resonator(L=Lload1, C=Cload1, freq=freq, name='load1')
    load2 =lc_resonator(L=Lload2, C=Cload2, freq=freq, name='load2')

    open = rf.Circuit.Open(freq, name='open')
    port50=rf.Circuit.Port(freq, name="P1", z0=50)

    cnx2 = [[(port50, 0), (cc_port, 0)],
       [(cc_port, 1), (load1, 0)],
        [(load1, 1), (cc1, 0)], 
        [(cc1, 1), (ctog1, 0)],
        [(ctog1, 1), (line, 0)],
        [(line, 1), (ctog2, 0)],
        [(ctog2, 1), (cc2, 0)],
        [(cc2, 1), (load2, 0)],
        [(load2, 1), (open, 0)]]
    
    ckt = rf.Circuit(cnx2, name="Port_Resonator")
    network_port = ckt.network
    return network_port


def resonances_from_res11(network, m=0, n=0):
    f = network.frequency.f / 1e9  # GHz
    s11 = network.s[:, m, n]
    peaks, _ = find_peaks(np.real(s11))
    resonance_freqs = f[peaks]
    #print("Resonance frequencies (GHz):", resonance_freqs)
    return resonance_freqs

def fit_Ceff_Leff(w0_guess, w0_window_frac, n_w0,
                             Cc1, Cc2, Z0, data_points, freq=None, k0=None, use_k = False, x0=None):
    """
    Scan w0 in a window around w0_guess.
    For each w0, do the existing 1D Nelder-Mead fit over Leff (Ceff is forced by resonance constraint).
    Return best (Ceff, Leff, w0_best) + some diagnostics.
    """

    dp = np.asarray(data_points, dtype=float)
    if dp.ndim != 2 or dp.shape[1] != 3:
        raise ValueError("data_points must be shape (N, 3): (omega, ReS11, ImS11)")

    freqs   = dp[:, 0]
    target  = dp[:, 1] + 1j*dp[:, 2]

    # Element impedances
    def ZCc1(w): return 1.0/(1j*w*Cc1)
    def ZCc2(w): return 1.0/(1j*w*Cc2)
    def ZL(w, Leff):    return 1j*w*Leff
    def ZC(w, Ceff):    return 1.0/(1j*w*Ceff)
    def ZLC(w, Leff, Ceff):
        return 1.0 / (1.0/ZL(w, Leff) + 1.0/ZC(w, Ceff))
    def Zshunt(w, Leff, Ceff):
        return 1.0 / (1.0/ZLC(w, Leff, Ceff) + 1.0/ZCc2(w))
    def Zin(w, Leff, Ceff):
        return ZCc1(w) + Zshunt(w, Leff, Ceff)
    def S11(w, Leff, Ceff):
        z = Zin(w, Leff, Ceff)
        return (z - Z0)/(z + Z0)

    # build w0 grid
    w0_min = w0_guess * (1.0 - w0_window_frac)
    w0_max = w0_guess * (1.0 + w0_window_frac)
    w0_grid = np.linspace(w0_min, w0_max, int(n_w0))

    best = None
    results = []

    for w0 in w0_grid:
        # resonance constraint: -1 + (Cc1 + Cc2 + Ceff) * Leff * w0^2 == 0
        def ceff_of_L(Leff):
            return 1.0/(Leff*(w0**2)) - (Cc1 + Cc2)

        # Leff bound from Ceff>0  => Leff < 1/((Cc1+Cc2)*w0^2)
        LeMax = 1.0 / ((Cc1 + Cc2) * (w0**2))
        Le_upper = 0.999 * LeMax

        def obj(Le):
            Le = float(Le)
            if not (0.0 < Le < Le_upper):
                dist = (0.0 - Le) if Le <= 0.0 else (Le - Le_upper)
                return 1e6*(1.0 + dist**2)

            Ceff = ceff_of_L(Le)
            if Ceff <= 0.0 or not np.isfinite(Ceff):
                return 1e6

            pred = np.array([S11(w, Le, Ceff) for w in freqs])
            resid = pred - target

            if use_k == True:

                sse = float(np.sum(resid.real**2 + resid.imag**2))
                
                if sse > 1e-3:   # tune this threshold
                    return sse
                
                lom_network =  lc_resonator_network(Cc1=Cc1, Cc2=Cc2, Leff=Le, Ceff=Ceff, freq=freq)

                k_pred = fwhm_from_res11(lom_network)
                #print(k_pred)   
                k_meas = k0

                rk = (k_pred - k_meas) / k_meas
                lam = 100.0  # tune
                #return float(np.sum(resid.real**2 + resid.imag**2)) + lam*(rk**2)

                return sse + lam*(rk*rk)

            else:
                resid = pred - target
                return float(np.sum(resid.real**2 + resid.imag**2))
            

        # choose an x0 that stays valid for this w0
        x0_local = 0.5 * Le_upper if x0 is None else min(float(x0), 0.9*Le_upper)

        fit = minimize(lambda x: obj(x[0]),
                       x0=np.array([x0_local], dtype=float),
                       method='Nelder-Mead',
                       options=dict(maxiter=3000, xatol=1e-30, fatol=1e-30, disp=False))

        LeffHat = float(fit.x[0])
        LeffHat = min(max(LeffHat, np.nextafter(0.0, 1.0)), Le_upper)
        CeffHat = ceff_of_L(LeffHat)
        sse = float(fit.fun)

        results.append((w0, CeffHat, LeffHat, sse))

        if best is None or sse < best[-1]:
            best = (w0, CeffHat, LeffHat, sse)

    w0_best, Ceff_best, Leff_best, sse_best = best
    return Ceff_best, Leff_best, w0_best, np.array(results)



def approximate_LOM_network(freq, data_ntw, Cc1, Cc2, Z0, n_dense=100, n_kappa = 0.75, k0=None, use_k = False, x0=None, w0_window_frac=0.005,
            n_w0=20):
        """Fit LOM parameters to data network."""

        data_points = build_sparse_data_points(
            data_ntw,
            n_dense=n_dense,
            n_kappa=n_kappa
        )
        w0 = resonance_from_res11(data_ntw) * 2 * np.pi
        #CeffHat, LeffHat = fit_Ceff_Leff_new(w0, Cc1, Cc2, Z0, data_points, x0=x0)
        CeffHat, LeffHat, w0_best, scan_results = fit_Ceff_Leff(
            w0_guess=w0,
            w0_window_frac=w0_window_frac,
            n_w0=n_w0,
            Cc1=Cc1,
            Cc2=Cc2,
            Z0=Z0,
            data_points=data_points,
            use_k = use_k,
            k0 = k0,
            x0=x0,
            freq=freq,
        )
        lom_network = lc_resonator_network(Cc1=Cc1, Cc2=Cc2, Leff=LeffHat, Ceff=CeffHat, freq=freq)
        f0_lom = resonance_from_res11(lom_network)
        linewidth_lom = fwhm_from_res11(lom_network)

        f0_cpw= resonance_from_res11(data_ntw)
        linewidth_cpw= fwhm_from_res11(data_ntw)
        print("Frequency (GHz):", f0_lom/ 1e9, " Linewidth (MHz):", linewidth_lom / 1e6)
        print("Frequency (Data) (GHz):", f0_cpw / 1e9, " Linewidth (Data) (MHz):", linewidth_cpw / 1e6)
        return lom_network, f0_cpw, f0_lom, linewidth_cpw, linewidth_lom, CeffHat, LeffHat                               