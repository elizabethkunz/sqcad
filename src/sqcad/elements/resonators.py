import numpy as np
import skrf as rf
from skrf.media import CPW
import numpy as np



class HalfWaveResonator(rf.Network):
    def __init__(self, freq, w, s, h, ep_r, f0=None, length=None,
        Z0=50, one_port=False):
        self.freq = freq
        self.Z0 = Z0
        self.one_port = one_port

        cpw = CPW(
            frequency=freq,
            w=w,
            s=s,
            h=h,
            ep_r=ep_r,
            z0=Z0
        )

        v_ph = cpw.v_p[0]

        if (f0 is None) and (length is None):
            raise ValueError("Must provide either f0 or length.")

        if (f0 is not None) and (length is not None):
            raise Warning("Both f0 and length provided. Using f0 to calculate length.")

        if f0 is not None:
            self.f0 = f0
            self.length = v_ph / (2 * f0)

        else:  # length provided
            self.length = length
            self.f0 = v_ph / (2 * length)


        line = cpw.line(self.length, unit='m')


        if one_port:
            open_term = rf.Circuit.Open(freq, z0=Z0, name="Open")
            network = line ** open_term
        else:
            network = line

        # Initialize parent Network with the FINAL network
        super().__init__(
            frequency=network.frequency,
            s=network.s,
            z0=network.z0
        )
        
    def summary(self):
            print("Half-Wave Resonator")
            print(f"Resonance target f0: {self.f0/1e9:.3f} GHz")
            print(f"Length: {self.length*1e3:.3f} mm")
            print(f"Characteristic impedance: {self.Z0} Ω")
        
        

class QuarterWaveResonator(rf.Network):
    """
    A wrapper around scikit-rf CPW to create a λ/4 resonator.
    
    Parameters
    ----------
    freq : skrf.Frequency
        Frequency object
    w : float
        Center conductor width (m)
    s : float
        Gap width (m)
    h : float
        Substrate height (m)
    ep_r : float
        Relative permittivity
    Z0 : float
        Reference impedance (default 50 Ω)
    f0 : float
        Desired resonance frequency (Hz)
    one_port : bool
        If True → shorted λ/4 (default)
        If False → open λ/4
    """
    def __init__(self, freq, w, s, h, ep_r, f0=None, length=None, Z0=50, one_port=True):

        self.freq = freq
        self.Z0 = Z0
        self.one_port = one_port

        cpw = CPW(
            frequency=freq,
            w=w,
            s=s,
            h=h,
            ep_r=ep_r,
            z0=Z0
        )

        v_ph = cpw.v_p[0]

        if (f0 is None) and (length is None):
            raise ValueError("Must provide either f0 or length.")

        if (f0 is not None) and (length is not None):
            raise Warning("Both f0 and length provided. Using f0 to calculate length.")

        if f0 is not None:
            self.f0 = f0
            self.length = v_ph / (4 * f0)

        else:  # length provided
            self.length = length
            self.f0 = v_ph / (4 * length)

        line = cpw.line(self.length, unit='m')

        if one_port:
            open_term = rf.Circuit.Open(freq, z0=Z0, name="Open")
            network = line ** open_term
        else:
            network = line

        # Initialize parent Network with the FINAL network
        super().__init__(
            frequency=network.frequency,
            s=network.s,
            z0=network.z0
        )

    def summary(self):
        print("Quarter-Wave Resonator")
        print(f"Resonance target f0: {self.f0/1e9:.3f} GHz")
        print(f"Length: {self.length*1e3:.3f} mm")
        print(f"Characteristic impedance: {self.Z0} Ω")