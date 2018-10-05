# -*- coding: utf-8 -*-
"""
Created on Wed Oct 03 18:37:32 2018

@author: calil
"""

from sharc.propagation.propagation import Propagation
from sharc.propagation.propagation_free_space_sf1395 import  PropagationFreeSpaceSf1395

import numpy as np


class PropagationHapsRas(Propagation):
    """
    Implements the Propagation from HAPS to RAS
    """
    def __init__(self, random_number_gen: np.random.RandomState):
        super().__init__(random_number_gen)

        self.propagaion_sf1395 = PropagationFreeSpaceSf1395(random_number_gen)

        self.attenuation = 0.5

    def get_loss(self, *args, **kwargs) -> np.array:

        d = kwargs["distance_3D"]

        f = kwargs["frequency"]

        elev = kwargs["elevation"]
        sat_par = kwargs["sat_params"]

        number_of_sectors = kwargs.pop("number_of_sectors", 1)

        loss_sf1359 = self.propagaion_sf1395.get_loss(distance_3D=d,
                                                      frequency=f,
                                                      elevation=elev,
                                                      sat_params=sat_par,
                                                      free_space_enable=False)

        loss = -1.0*self.attenuation + 10*np.log10(4*np.pi*np.power(d, 2)) + loss_sf1359

        if number_of_sectors > 1:
            loss = np.repeat(loss, number_of_sectors, 1)

        return loss
