# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 15:04:09 2018

@author: Calil
"""

import unittest
import numpy as np
import numpy.testing as npt
import math

from sharc.simulation_uplink import SimulationUplink
from sharc.parameters.parameters import Parameters
from sharc.antenna.antenna_omni import AntennaOmni
from sharc.antenna.antenna_beamforming_imt import AntennaBeamformingImt
from sharc.station_factory import StationFactory
from sharc.propagation.propagation_factory import PropagationFactory

class SimulationUplinkTest(unittest.TestCase):

    def setUp(self):
        self.param = Parameters()

        self.param.general.imt_link = "UPLINK"
        self.param.general.enable_cochannel = True
        self.param.general.enable_adjacent_channel = False
        self.param.general.overwrite_output = True

        self.param.imt.topology = "SINGLE_BS"
        self.param.imt.num_macrocell_sites = 19
        self.param.imt.num_clusters = 2
        self.param.imt.intersite_distance = 150
        self.param.imt.minimum_separation_distance_bs_ue = 0
        self.param.imt.interfered_with = False
        self.param.imt.frequency = 38750
        self.param.imt.bandwidth = 1428.6
        self.param.imt.rb_bandwidth = 0.180
        self.param.imt.spectral_mask = "ITU 265-E"
        self.param.imt.guard_band_ratio = 0.0
        self.param.imt.bs_load_probability = 1
        self.param.imt.bs_conducted_power = 20
        self.param.imt.bs_height = 20000
        self.param.imt.bs_noise_figure = 12
        self.param.imt.bs_noise_temperature = 290
        self.param.imt.bs_ohmic_loss = 3
        self.param.imt.ul_attenuation_factor = 0.4
        self.param.imt.ul_sinr_min = -10
        self.param.imt.ul_sinr_max = 22
        self.param.imt.ue_k = 2
        self.param.imt.ue_k_m = 1
        self.param.imt.ue_indoor_percent = 0
        self.param.imt.ue_distribution_distance = "UNIFORM"
        self.param.imt.ue_distribution_azimuth = "UNIFORM"
        self.param.imt.ue_distribution_type = "ANGLE_AND_DISTANCE"
        self.param.imt.ue_tx_power_control = "OFF"
        self.param.imt.ue_p_o_pusch = -95
        self.param.imt.ue_alpha = 1
        self.param.imt.ue_p_cmax = 20
        self.param.imt.ue_conducted_power = 10
        self.param.imt.ue_height = 10
        self.param.imt.ue_noise_figure = 12
        self.param.imt.ue_ohmic_loss = 1.5
        self.param.imt.ue_body_loss = 0
        self.param.imt.dl_attenuation_factor = 0.6
        self.param.imt.dl_sinr_min = -10
        self.param.imt.dl_sinr_max = 30
        self.param.imt.channel_model = "FSPL"
        self.param.imt.line_of_sight_prob = 0.95 # probability of line-of-sight (not for FSPL)
        self.param.imt.shadowing = False
        self.param.imt.noise_temperature = 290
        self.param.imt.BOLTZMANN_CONSTANT = 1.38064852e-23
        
        self.param.antenna_imt.ue_antenna_type = "F1245"
        self.param.antenna_imt.peak_gain = 50
        self.param.antenna_imt.diameter = 2
        self.param.antenna_imt.normalization = False
        self.param.antenna_imt.bs_element_pattern = "M2101"
        self.param.antenna_imt.bs_normalization_file = None
        self.param.antenna_imt.bs_tx_element_max_g = 5
        self.param.antenna_imt.bs_tx_element_phi_3db = 65
        self.param.antenna_imt.bs_tx_element_theta_3db = 65
        self.param.antenna_imt.bs_tx_element_am = 30
        self.param.antenna_imt.bs_tx_element_sla_v = 30
        self.param.antenna_imt.bs_tx_n_rows = 8
        self.param.antenna_imt.bs_tx_n_columns = 8
        self.param.antenna_imt.bs_tx_element_horiz_spacing = 0.5
        self.param.antenna_imt.bs_tx_element_vert_spacing = 0.5
        self.param.antenna_imt.bs_rx_element_max_g = 5
        self.param.antenna_imt.bs_rx_element_phi_deg_3db = 65
        self.param.antenna_imt.bs_rx_element_theta_deg_3db = 65
        self.param.antenna_imt.bs_rx_element_am = 30
        self.param.antenna_imt.bs_rx_element_sla_v = 30
        self.param.antenna_imt.bs_rx_n_rows = 8
        self.param.antenna_imt.bs_rx_n_columns = 8
        self.param.antenna_imt.bs_rx_element_horiz_spacing = 0.5
        self.param.antenna_imt.bs_rx_element_vert_spacing = 0.5
        self.param.antenna_imt.bs_downtilt_deg = 10
        self.param.antenna_imt.ue_element_pattern = "M2101"
        self.param.antenna_imt.ue_normalization_file = None
        self.param.antenna_imt.ue_tx_element_max_g = 5
        self.param.antenna_imt.ue_tx_element_phi_deg_3db = 90
        self.param.antenna_imt.ue_tx_element_theta_deg_3db = 90
        self.param.antenna_imt.ue_tx_element_am = 25
        self.param.antenna_imt.ue_tx_element_sla_v = 25
        self.param.antenna_imt.ue_tx_n_rows = 4
        self.param.antenna_imt.ue_tx_n_columns = 4
        self.param.antenna_imt.ue_tx_element_horiz_spacing = 0.5
        self.param.antenna_imt.ue_tx_element_vert_spacing = 0.5
        self.param.antenna_imt.ue_rx_element_max_g = 5
        self.param.antenna_imt.ue_rx_element_phi_3db = 90
        self.param.antenna_imt.ue_rx_element_theta_3db = 90
        self.param.antenna_imt.ue_rx_element_am = 25
        self.param.antenna_imt.ue_rx_element_sla_v = 25
        self.param.antenna_imt.ue_rx_n_rows = 4
        self.param.antenna_imt.ue_rx_n_columns = 4
        self.param.antenna_imt.ue_rx_element_horiz_spacing = 1
        self.param.antenna_imt.ue_rx_element_vert_spacing = 1

        self.param.fss_es.x = -5000
        self.param.fss_es.y = 0
        self.param.fss_es.location = "FIXED"
        self.param.fss_es.height = 10
        self.param.fss_es.elevation_min = 20
        self.param.fss_es.elevation_max = 20
        self.param.fss_es.azimuth = "0"
        self.param.fss_es.frequency = 38750
        self.param.fss_es.bandwidth = 200
        self.param.fss_es.noise_temperature = 150
        self.param.fss_es.tx_power_density = -31.1
        self.param.fss_es.antenna_gain = 50
        self.param.fss_es.antenna_pattern = "OMNI"
        self.param.fss_es.channel_model = "FSPL"
        self.param.fss_es.line_of_sight_prob = 1
        self.param.fss_es.acs = 0
        self.param.fss_es.BOLTZMANN_CONSTANT = 1.38064852e-23
        self.param.fss_es.EARTH_RADIUS = 6371000


    def test_simulation_2bs_4ue_es(self):
        self.param.general.system = "FSS_ES"

        self.simulation = SimulationUplink(self.param, "")
        self.simulation.initialize()

        self.simulation.bs_power_gain = 0
        self.simulation.ue_power_gain = 0

        random_number_gen = np.random.RandomState()

        self.simulation.bs = StationFactory.generate_imt_base_stations(self.param.imt,
                                                                       self.param.antenna_imt,
                                                                       self.simulation.topology,
                                                                       random_number_gen)
        self.simulation.bs.antenna = np.array([AntennaOmni(1), AntennaOmni(2)])
        self.simulation.bs.active = np.ones(2, dtype=bool)

        self.simulation.ue = StationFactory.generate_imt_ue(self.param.imt,
                                                            self.param.antenna_imt,
                                                            self.simulation.topology,
                                                            random_number_gen)
        self.simulation.ue.x = np.array([10, 50, 100, 140])
        self.simulation.ue.y = np.array([ 0,  0,   0,   0])
        
        self.simulation.ue.active = np.ones(4, dtype=bool)

        # test connection method
        self.simulation.connect_ue_to_bs()
        self.simulation.select_ue(random_number_gen)
        
        # Test gains
        gain = self.simulation.calculate_gains(self.simulation.ue,
                                               self.simulation.bs)
        npt.assert_allclose(gain,
                            np.array([[ 50.00, 35.04],
                                      [ 50.00, 35.04],
                                      [ 35.04, 50.00],
                                      [ 35.04, 50.00]]),
                            atol=1e-2)
    
    def test_simulation_1bs_4ue_es(self):
        self.param.imt.num_clusters = 1
        self.param.imt.ue_k = 4
        self.param.imt.ue_k_m = 1
        
        self.param.general.system = "FSS_ES"

        self.simulation = SimulationUplink(self.param, "")
        self.simulation.initialize()

        self.simulation.bs_power_gain = 0
        self.simulation.ue_power_gain = 0

        random_number_gen = np.random.RandomState()

        self.simulation.bs = StationFactory.generate_imt_base_stations(self.param.imt,
                                                                       self.param.antenna_imt,
                                                                       self.simulation.topology,
                                                                       random_number_gen)
        self.simulation.bs.antenna = np.array([AntennaOmni(1)])
        self.simulation.bs.active = np.ones(1, dtype=bool)

        self.simulation.ue = StationFactory.generate_imt_ue(self.param.imt,
                                                            self.param.antenna_imt,
                                                            self.simulation.topology,
                                                            random_number_gen)
        self.simulation.ue.x = np.array([ 100,    0,-100,   0])
        self.simulation.ue.y = np.array([   0,  100,   0,-100])
        
        self.simulation.ue.active = np.ones(4, dtype=bool)

        # test connection method
        self.simulation.connect_ue_to_bs()
        self.simulation.select_ue(random_number_gen)
        
        # Test gains
        gain = self.simulation.calculate_gains(self.simulation.ue,
                                               self.simulation.bs)
        npt.assert_allclose(gain,
                            np.array([[50.00],
                                      [50.00],
                                      [50.00],
                                      [50.00]]),
                            atol=1e-2)
        

if __name__ == '__main__':
    unittest.main()

