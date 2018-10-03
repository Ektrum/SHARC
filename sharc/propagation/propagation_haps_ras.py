# -*- coding: utf-8 -*-
"""
Created on Wed Oct 03 18:37:32 2018

@author: calil
"""

from sharc.propagation.propagation import Propagation

import numpy as np


class PropagationHapsRas(Propagation):
    """
    Implements the Propagation from HAPS to RAS
    """
    def __init__(self, random_number_gen: np.random.RandomState):
        super().__init__(random_number_gen)

        self.attenuation = 0.5

    def get_loss(self, *args, **kwargs) -> np.array:

        if "distance_2D" in kwargs:
            d = kwargs["distance_2D"]
        else:
            d = kwargs["distance_3D"]

        number_of_sectors = kwargs.pop("number_of_sectors",1)

        loss = -1.0*self.attenuation + 10*np.log10(4*np.pi*d)

        if number_of_sectors > 1:
            loss = np.repeat(loss, number_of_sectors, 1)

        return loss
