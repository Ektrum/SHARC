# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 16:41:25 2017

@author: edgar
"""

from topology import Topology
from topology_macrocell import TopologyMacrocell
from topology_single_base_station import TopologySingleBaseStation
from parameters.parameters_imt import ParametersImt

class TopologyFactory(object):
    
    @staticmethod
    def createTopology(param: ParametersImt) -> Topology:
        if param.topology == "SINGLE_BS":
            return TopologySingleBaseStation(param.intersite_distance/2, param.num_clusters)
        elif param.topology == "MACROCELL":
            return TopologyMacrocell(param.intersite_distance, param.num_clusters)
