# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 12:04:27 2017

@author: edgar
"""

from sharc.propagation.propagation import Propagation
from sharc.propagation.propagation_free_space import PropagationFreeSpace
from sharc.propagation.propagation_clutter_loss import PropagationClutterLoss
from sharc.support.enumerations import StationType

import numpy as np

class PropagationFreeSpaceSf1395(Propagation):
    """
    Implements the simplified satellite propagation model
    """

    def __init__(self, random_number_gen: np.random.RandomState):
        super().__init__(random_number_gen)
        self.free_space = PropagationFreeSpace(random_number_gen)


    def get_loss(self, *args, **kwargs) -> np.array:
        d = kwargs["distance_3D"]
        f = kwargs["frequency"]
        f = np.unique(f)

        elevation = kwargs["elevation"]
        sat_params = kwargs["sat_params"]
        h = sat_params.imt_altitude/1000

        number_of_sectors = kwargs.pop("number_of_sectors", 1)

        free_space_enable = kwargs.pop("free_space_enable",
                                       True)

        if free_space_enable:
            free_space_loss = self.free_space.get_loss(distance_3D=d, frequency=f)
        else:
            free_space_loss = 0.0

        elevation = np.array(elevation['free_space'])
        negative_angles = np.nonzero(elevation < 0)
        elevation[negative_angles] = 0

        pfd = kwargs.pop('pfd',False)

        if f < 10700:
            error_message = "SF1395 model does not support frequencies below 10.7 GHz"
            raise ValueError(error_message)

        # calculate atmospheric gases loss
        if abs(sat_params.imt_lat_deg) < 22.5:
            # low-latitude
            if f <= 11700:
                att = 3.4 / (1 + 0.8356*elevation + h*(0.2693+0.2753*elevation) + 0.1002*h**2)
            elif f <= 12750:
                att = 3.84 / (1 + 0.8598 * elevation + h * (0.2815 + 0.3031 * elevation) + 0.1148 * h ** 2)
            elif f <= 14300:
                att_low = 3.84 / (1 + 0.8598 * elevation + h * (0.2815 + 0.3031 * elevation) + 0.1148 * h ** 2)
                att_high = 5.59 / (1 + 0.9245 * elevation + h * (0.3063 + 0.3929 * elevation) + 0.1671 * h ** 2)
                att = att_low + (att_high - att_low) * (f-12750)/(14300-12750)
            elif f <= 14800:
                att = 5.59 / (1 + 0.9245 * elevation + h * (0.3063 + 0.3929 * elevation) + 0.1671 * h**2)
            elif f <= 17700:
                att_low = 5.59 / (1 + 0.9245 * elevation + h * (0.3063 + 0.3929 * elevation) + 0.1671 * h**2)
                att_high = 11.38/(1 + 0.8601*elevation + 0.04510*elevation**2 + h*(0.2342 + 0.6585*elevation) +
                                  0.2658*h**2)
                att = att_low + (att_high - att_low) * (f-14800)/(17700-14800)
            elif f <= 18800:
                att = 11.38/(1 + 0.8601*elevation + 0.04510 * elevation**2 + h*(0.2342 + 0.6585*elevation) +
                             0.2658*h**2)
            elif f <= 19300:
                att = 16.17 / (1 + 0.9205 * elevation + 0.03829 * elevation**2 + h * (0.2888 + 0.4380 * elevation) +
                               h**2 * (0.2481 + 0.1380 * elevation))
            elif f <= 19700:
                att = 19.17 / (1 + 0.9089 * elevation + 0.04175 * elevation**2 + h * (0.2674 + 0.4401 * elevation) +
                               h**2 * (0.2570 + 0.1485 * elevation))
            elif f <= 27000:
                att_low = 19.17 / (1 + 0.9089 * elevation + 0.04175 * elevation**2 + h * (0.2674 + 0.4401 * elevation) +
                                   h**2 * (0.2570 + 0.1485 * elevation))
                att_high = 22.73 / (1 + 0.9463 * elevation + 0.03455 * elevation**2 + h * (0.3232 + 0.4519 * elevation) +
                                    h**2 * (0.2486 + 0.1317 * elevation))
                att = att_low + (att_high - att_low) * (f-19700)/(27000-19700)
            elif f <= 27500:
                att = 22.73 / (1 + 0.9463 * elevation + 0.03455 * elevation**2 + h * (0.3232 + 0.4519 * elevation) +
                               h**2 * (0.2486 + 0.1317 * elevation))
            elif f <= 29500:
                att = 20.10 / (1 + 0.9428 * elevation + 0.02816 * elevation**2 + h * (0.3417 + 0.4499 * elevation) +
                               h**2 * (0.2165 + 0.09728 * elevation))
            elif f <= 37500:
                att_low = 20.10 / (1 + 0.9428 * elevation + 0.02816 * elevation**2 + h * (0.3417 + 0.4499 * elevation) +
                                   h**2 * (0.2165 + 0.09728 * elevation))
                att_high = 23.21 / (1 + 0.8042 * elevation + 0.05421 * elevation**2 - 0.001771 * elevation**3
                                    + .1382e-4 * elevation**4 + h * (0.2743 + 0.4897 * elevation) + 0.1742 * h**2)
                att = att_low + (att_high - att_low) * (f-29500)/(37500-29500)
            elif f <= 40500:
                att = 23.21 / (1 + 0.8042 * elevation + 0.05421 * elevation**2 - 0.001771 * elevation**3
                               + .1382e-4 * elevation**4 + h * (0.2743 + 0.4897 * elevation) + 0.1742 * h**2)
            elif f <= 42500:
                att = 27.78 / (1 + 0.7880 * elevation + 0.04877 * elevation**2 - 0.001566 * elevation**3
                               + .1202e-4 * elevation**4 + h * (0.2729 + 0.4361 * elevation) + 0.1473 * h**2)
            elif f <= 43500:
                att = 32.19 / (1 + 0.7732 * elevation + 0.04549 * elevation**2 - 0.001445 * elevation**3
                               + .1096e-4 * elevation ** 4 + h * (0.2687 + 0.3992 * elevation) + 0.1297 * h**2)
            elif f <= 47200:
                att_low = 32.19 / (1 + 0.7732 * elevation + 0.04549 * elevation**2 - 0.001445 * elevation**3
                                   + .1096e-4 * elevation**4 + h * (0.2687 + 0.3992 * elevation) + 0.1297 * h ** 2)
                att_high = 52.43 / (1 + 0.7364 * elevation + 0.03601 * elevation**2 - 0.001099 * elevation**3
                                    + .8024e-5 * elevation**4 + h * (0.2642 + 0.2479 * elevation)
                                    + h**2 * (0.08130 + 0.02637 * elevation))
                att = att_low + (att_high - att_low) * (f - 43500) / (47200 - 43500)
            elif f <= 50200 and not (47900 < f <= 48200):
                att = 52.43 / (1 + 0.7364 * elevation + 0.03601 * elevation**2 - 0.001099 * elevation**3
                               + .8024e-5 * elevation**4 + h * (0.2642 + 0.2479 * elevation)
                               + h**2 * (0.08130 + 0.02637 * elevation))
            elif 47900 < f <= 48200:
                att = 57.90 / (1 + 0.7262 * elevation + 0.03534 * elevation**2 - 0.001074 * elevation**3
                               + .7826e-5 * elevation**4 + h * (0.2576 + 0.2382 * elevation)
                               + h**2 * (0.07645 + 0.02443 * elevation))
            else:
                error_message = "SF1395 model does not support frequencies above 50.2 GHz"
                raise ValueError(error_message)
        elif abs(sat_params.imt_lat_deg) < 45:
            # mid-latitude
            if f <= 11700:
                att = 3.01 / (1 + 0.7509*elevation + h * (0.3991+0.2149*elevation))
            elif f <= 12750:
                att = 3.23 / (1 + 0.7585 * elevation + h * (0.4154 + 0.2232 * elevation))
            elif f <= 14300:
                att_low = 3.23 / (1 + 0.7585 * elevation + h * (0.4154 + 0.2232 * elevation))
                att_high = 4.00 / (1 + 0.8411 * elevation + h * (0.2844 + 0.2832 * elevation) + 0.09301 * h**2)
                att = att_low + (att_high - att_low) * (f - 12750) / (14300 - 12750)
            elif f <= 14800:
                att = 4.00 / (1 + 0.8411 * elevation + h * (0.2844 + 0.2832 * elevation) + 0.09301 * h**2)
            elif f <= 17700:
                att_low = 4.00 / (1 + 0.8411 * elevation + h * (0.2844 + 0.2832 * elevation) + 0.09301 * h**2)
                att_high = 6.54 / (1 + 0.8994 * elevation + h * (0.2971 + 0.3762 * elevation) + 0.1322 * h**2)
                att = att_low + (att_high - att_low) * (f-14800)/(17700-14800)
            elif f <= 18800:
                att = 6.54 / (1 + 0.8994 * elevation + h * (0.2971 + 0.3762 * elevation) + 0.1322 * h**2)
            elif f <= 19300:
                att = 8.38 / (1 + 0.9117 * elevation + h * (0.2821 + 0.4201 * elevation) + 0.1500 * h ** 2)
            elif f <= 19700:
                att = 9.34 / (1 + 0.7790 * elevation + 0.03929 * elevation**2 + h * (0.2256 + 0.4979 * elevation)
                              + 0.1562 * h ** 2)
            elif f <= 27000:
                att_low = 9.34 / (1 + 0.7790 * elevation + 0.03929 * elevation**2 + h * (0.2256 + 0.4979 * elevation)
                              + 0.1562 * h ** 2)
                att_high = 11.96 / (1 + 0.8121 * elevation + 0.03055 * elevation**2 + h * (0.2619 + 0.4728 * elevation) +
                                    h**2 * 0.1490)
                att = att_low + (att_high - att_low) * (f - 19700) / (27000 - 19700)
            elif f <= 27500:
                att = 11.96 / (1 + 0.8121 * elevation + 0.03055 * elevation**2 + h * (0.2619 + 0.4728 * elevation) +
                               h**2 * 0.1490)
            elif f <= 29500:
                att = 11.51 / (1 + 0.8174 * elevation + 0.02298 * elevation**2 + h * (0.2734 + 0.4214 * elevation) +
                               h**2 * 0.1291)
            elif f <= 37500:
                att_low = 11.51 / (1 + 0.8174 * elevation + 0.02298 * elevation**2 + h * (0.2734 + 0.4214 * elevation) +
                               h**2 * 0.1291)
                att_high = 16.60 / (1 + 0.8121 * elevation + 0.01302 * elevation**2 + h * (0.3027 + 0.2572 * elevation) +
                               h**2 * (0.07186 + 0.03217 * elevation))
                att = att_low + (att_high - att_low) * (f - 29500) / (37500 - 29500)
            elif f <= 40500:
                att = 16.60 / (1 + 0.8121 * elevation + 0.01302 * elevation**2 + h * (0.3027 + 0.2572 * elevation) +
                               h**2 * (0.07186 + 0.03217 * elevation))
            elif f <= 42500:
                att = 20.76 / (1 + 0.6980 * elevation + 0.04731 * elevation**2 - 0.001508 * elevation**3
                               + .1157e-4 * elevation**4 + h * (0.2497 + 0.3257 * elevation) + 0.07995 * h**2)
            elif f <= 43500:
                att = 25.20 / (1 + 0.6884 * elevation + 0.04608 * elevation**2 - 0.001462 * elevation**3
                           + .1117e-4 * elevation**4 + h * (0.2437 + 0.3107 * elevation) + 0.07470 * h**2)
            elif f <= 47200:
                att_low = 25.20 / (1 + 0.6884 * elevation + 0.04608 * elevation**2 - 0.001462 * elevation**3
                                   + .1117e-4 * elevation**4 + h * (0.2437 + 0.3107 * elevation) + 0.07470 * h**2)
                att_high = 47.00 / (1 + 0.7004 * elevation + 0.03568 * elevation**2 - 0.001081 * elevation**3
                                    + .7878e-5 * elevation**4 + h * (0.2527 + 0.1970 * elevation)
                                    + h**2 * (0.05539 + 0.03239 * elevation))
                att = att_low + (att_high - att_low) * (f - 43500) / (47200 - 43500)
            elif f <= 50200 and not (47900 < f <= 48200):
                att = 47.00 / (1 + 0.7004 * elevation + 0.03568 * elevation**2 - 0.001081 * elevation**3
                               + .7878e-5 * elevation**4 + h * (0.2527 + 0.1970 * elevation)
                               + h**2 * (0.05539 + 0.03239 * elevation))
            elif 47900 < f <= 48200:
                att = 53.06 / (1 + 0.6962 * elevation + 0.03555 * elevation**2 - 0.001076 * elevation**3
                               + .7840e-5 * elevation**4 + h * (0.2495 + 0.1940 * elevation)
                               + h**2 * (0.05420 + 0.03176 * elevation))
            else:
                error_message = "SF1395 model does not support frequencies above 50.2 GHz"
                raise ValueError(error_message)

        else:
            # high-latitude
            if f < 11700:
                att = 2.98 / (1 + 0.7477*elevation + h*(0.3737+0.2072*elevation))
            elif f < 12750:
                att = 3.12 / (1 + 0.7487 * elevation + h * (0.3792 + 0.2102 * elevation))
            elif f < 14300:
                att_low = 3.12 / (1 + 0.7487 * elevation + h * (0.3792 + 0.2102 * elevation))
                att_high = 3.63 / (1 + 0.7509 * elevation + h * (0.3973 + 0.2205 * elevation))
                att = att_low + (att_high - att_low) * (f - 12750) / (14300 - 12750)
            elif f < 14800:
                att = 3.63 / (1 + 0.7509 * elevation + h * (0.3973 + 0.2205 * elevation))
            elif f < 17700:
                att_low = 3.63 / (1 + 0.7509 * elevation + h * (0.3973 + 0.2205 * elevation))
                att_high = 4.95 / (1 + 0.8149 * elevation + h * (0.2205 + 0.2830 * elevation) + 0.09616 * h ** 2)
                att = att_low + (att_high - att_low) * (f-14800)/(17700-14800)
            elif f < 18800:
                att = 4.95 / (1 + 0.8149 * elevation + h * (0.2205 + 0.2830 * elevation) + 0.09616 * h ** 2)
            elif f < 19300:
                att = 5.87 / (1 + 0.8171 * elevation + h * (0.1962 + 0.3061 * elevation) + 0.1079 * h ** 2)
            elif f < 19700:
                att = 6.45 / (1 + 0.8152 * elevation + h * (0.1799 + 0.3163 * elevation) + 0.1141 * h ** 2)
            elif f < 27000:
                att_low = 6.45 / (1 + 0.8152 * elevation + h * (0.1799 + 0.3163 * elevation) + 0.1141 * h ** 2)
                att_high = 8.77 / (1 + 0.8259 * elevation + h * (0.2163 + 0.3037 * elevation) + h**2 * 0.1067)
                att = att_low + (att_high - att_low) * (f - 19700) / (27000 - 19700)
            elif f < 27500:
                att = 8.77 / (1 + 0.8259 * elevation + h * (0.2163 + 0.3037 * elevation) + h**2 * 0.1067)
            elif f < 29500:
                att = 9.00 / (1 + 0.8202 * elevation + h * (0.2324 + 0.2825 * elevation) + h**2 * 0.09510)
            elif f < 37500:
                att_low = 9.00 / (1 + 0.8202 * elevation + h * (0.2324 + 0.2825 * elevation) + h**2 * 0.09510)
                att_high = 14.44 / (1 + 0.7365 * elevation + 0.01542 * elevation**2 + h * (0.2202 + 0.2754 * elevation)
                                    + h**2 * 0.07416)
                att = att_low + (att_high - att_low) * (f - 29500) / (37500 - 29500)
            elif f < 40500:
                att = 14.44 / (1 + 0.7365 * elevation + 0.01542 * elevation**2 + h * (0.2202 + 0.2754 * elevation)
                               + h**2 * 0.07416)
            elif f < 42500:
                att = 18.92 / (1 + 0.6577 * elevation + 0.04678 * elevation**2 - 0.001484 * elevation**3
                               + .1139e-4 * elevation**4 + h * (0.2200 + 0.2811 * elevation) + 0.06507 * h**2)
            elif f < 43500:
                att = 23.56 / (1 + 0.6557 * elevation + 0.04605 * elevation**2 - 0.001457 * elevation**3
                               + .1115e-4 * elevation**4 + h * (0.2216 + 0.2749 * elevation) + 0.06237 * h**2)
            elif f <= 47200:
                att_low = 23.56 / (1 + 0.6557 * elevation + 0.04605 * elevation**2 - 0.001457 * elevation**3
                                   + .1115e-4 * elevation**4 + h * (0.2216 + 0.2749 * elevation) + 0.06237 * h**2)
                att_high = 46.70 / (1 + 0.6872 * elevation + 0.03637 * elevation**2 - 0.001105 * elevation**3
                                    + .8087e-5 * elevation**4 + h * (0.2472 + 0.1819 * elevation)
                                    + h**2 * (0.04858 + 0.03221 * elevation))
                att = att_low + (att_high - att_low) * (f - 43500) / (47200 - 43500)
            elif f <= 50200 and not (47900 < f <= 48200):
                att = 46.70 / (1 + 0.6872 * elevation + 0.03637 * elevation**2 - 0.001105 * elevation**3
                               + .8087e-5 * elevation**4 + h * (0.2472 + 0.1819 * elevation)
                               + h**2 * (0.04858 + 0.03221 * elevation))
            elif 47900 < f <= 48200:
                att = 53.21 / (1 + 0.6864 * elevation + 0.03632 * elevation**2 - 0.001103 * elevation**3
                               + .8073e-5 * elevation**4 + h * (0.2476 + 0.1812 * elevation)
                               + h**2 * (0.04791 + 0.03191 * elevation))
            else:
                error_message = "SF1395 model does not support frequencies above 50.2 GHz"
                raise ValueError(error_message)

        loss = free_space_loss + att

        if pfd:
            loss = loss + 10*np.log10(4*np.pi*np.power(d, 2))

        if number_of_sectors > 1:
            loss = np.repeat(loss, number_of_sectors, 1)

        return loss
