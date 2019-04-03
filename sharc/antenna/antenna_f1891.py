# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 12:24:13 2017

@author: edgar
"""

from sharc.antenna.antenna import Antenna
from sharc.parameters.parameters_antenna_imt import ParametersAntennaImt

import numpy as np

class AntennaF1891(Antenna):
    """
    Implements the antenna radiation pattern described in ITU-R F.1891. This is
    a phased array antenna that is described in, and that complies with,
    Resolution 221 (Rev.WRC-07) and can be used in both the HAPS gateway
    (ground) station and in the HAPS (airborne) platform.
    """

    def __init__(self, param: ParametersAntennaImt):
        super().__init__()
        self.peak_gain = param.peak_gain
        self.psi_b = np.sqrt(7442/np.power(10, 0.1 * self.peak_gain))
        self.l_n = param.antenna_l_n
        self.l_f = self.peak_gain - 73
        self.psi_1 = self.psi_b * np.sqrt(-self.l_n/3)
        self.psi_2 = 3.745 * self.psi_b
        self.x = self.peak_gain + self.l_n + 60*np.log10(self.psi_2)
        self.psi_3 = np.power(10, (self.x - self.l_f)/60)

    def add_beam(self, phi: float, theta: float):
        self.beams_list.append((phi, theta))

    def calculate_gain(self, *args, **kwargs) -> np.array:
        phi_vec = np.array(kwargs["phi_vec"])
        theta_vec = np.array(kwargs["theta_vec"])
        beams_l = np.array(kwargs["beams_l"])

        psi = self.calculate_off_axis_angle(phi_vec, theta_vec, beams_l)

        gain = self.l_f * np.ones(len(psi))

        idx_0 = np.where(psi <= self.psi_1)[0]
        gain[idx_0] = self.peak_gain - 3*np.power(psi[idx_0]/self.psi_b, 2)

        idx_1 = np.where((self.psi_1 < psi) & (psi <= self.psi_2))[0]
        gain[idx_1] = self.peak_gain + self.l_n

        idx_2 = np.where((self.psi_2 < psi) & (psi <= self.psi_3))[0]
        gain[idx_2] = self.x - 60*np.log10(psi[idx_2])

        idx_max_gain = np.where(beams_l == -1)[0]
        gain[idx_max_gain] = self.peak_gain

        return gain

    def calculate_off_axis_angle(self, Az, b, beams):
        Az0_list = np.array(self.beams_list)[beams.astype(int)]
        Az0 = np.array([k[0] for k in Az0_list])

        a_list = np.array(self.beams_list)[beams.astype(int)]
        a = np.array([k[1] for k in a_list])
        C = Az0 - Az

        off_axis_cos = np.cos(np.radians(a)) * np.cos(np.radians(b)) \
                       + np.sin(np.radians(a)) * np.sin(np.radians(b)) * np.cos(np.radians(C))
        off_axis_cos[np.where(off_axis_cos > 1)] = 1.0
        off_axis_rad = np.arccos(off_axis_cos)
        off_axis_deg = np.degrees(off_axis_rad)

        return off_axis_deg


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # initialize antenna parameters
    param = ParametersAntennaImt()
    param.antenna_pattern = "ITU-R F.1891"
    param.antenna_l_n = -25
    param.peak_gain = 28.1
    phi = np.linspace(-180,180, num = 10001)
    the = np.zeros_like(phi)
    beams = np.zeros_like(phi, dtype=int)

    antenna = AntennaF1891(param)
    antenna.add_beam(0.0, 0.0)
    gain = antenna.calculate_gain(phi_vec=the,
                                  theta_vec=phi,
                                  beams_l=beams)

    fig = plt.figure(figsize=(7,5), facecolor='w', edgecolor='k')  # create a figure object
    ax1 = fig.add_subplot(111)

    ax1.plot(phi, gain, "-b", label="$G_m = 28.1$ dB")
    ax1.set_ylim((-30, 30))
    ax1.set_xlim((-180, 180))
    ax1.set_title("ITU-R F.1891 antenna radiation pattern")
    ax1.set_xlabel("Off-axis angle [deg]")
    ax1.set_ylabel("Gain [dBi]")
    #ax1.legend(loc="upper right")
    #ax1.set_yticks([-40, -20, 0, 20, 40, 60])
    #ax1.set_xticks(np.linspace(0, 20, 11).tolist())
    ax1.grid(True)

    plt.show()
