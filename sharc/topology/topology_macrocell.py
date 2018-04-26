# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 12:51:22 2017

@author: edgar
"""

from sharc.topology.topology import Topology
import matplotlib.pyplot as plt
import matplotlib.axes

import math
import numpy as np

class TopologyMacrocell(Topology):
    """
    Generates the coordinates of the sites based on the macrocell network
    topology.
    """

    # possible values for base station azimuth and elevation [degrees]
    AZIMUTH = [60, 180, 300]
    ELEVATION = -10

    ALLOWED_NUM_CLUSTERS = [1, 7]

    def __init__(self, intersite_distance: float, num_clusters: int, number_of_sectors = 3,
                 beamwidth = -1, beams_dist = 0, bs_height = 0):
        """
        Constructor method that sets the parameters and already calls the
        calculation methods.

        Parameters
        ----------
            intersite_distance : Distance between two sites
            num_clusters : Number of clusters, should be 1 or 7
        """
        if num_clusters not in TopologyMacrocell.ALLOWED_NUM_CLUSTERS:
            error_message = "invalid number of clusters ({})".format(num_clusters)
            raise ValueError(error_message)

        if number_of_sectors not in [1, 3] and beamwidth < 0:
            error_message = "invalid number of sectors ({})".format(num_clusters)
            raise ValueError(error_message)

        if number_of_sectors == 1:
            beamwidth = -1

        cell_radius = intersite_distance*2/3
        super().__init__(intersite_distance, cell_radius)
        self.num_clusters = num_clusters
        self.num_sectors = number_of_sectors
        self.beamwidth = beamwidth
        self.beams_dist = beams_dist
        self.height = bs_height

    def calculate_coordinates(self, random_number_gen=np.random.RandomState()):
        """
        Calculates the coordinates of the stations according to the inter-site
        distance parameter. This method is invoked in all snapshots but it can
        be called only once for the macro cell topology. So we set
        static_base_stations to True to avoid unnecessary calculations.
        """
        if not self.static_base_stations:

            #self.static_base_stations = True

            d = self.intersite_distance
            h = (d/3)*math.sqrt(3)/2

            # these are the coordinates of the central cluster
            x_central = np.array([0, d, d/2, -d/2, -d, -d/2,
                             d/2, 2*d, 3*d/2, d, 0, -d,
                             -3*d/2, -2*d, -3*d/2, -d, 0, d, 3*d/2])
            y_central = np.array([0, 0, 3*h, 3*h, 0, -3*h,
                             -3*h, 0, 3*h, 6*h, 6*h, 6*h,
                             3*h, 0, -3*h, -6*h, -6*h, -6*h, -3*h])
            self.x = np.copy(x_central)
            self.y = np.copy(y_central)

            # other clusters are calculated by shifting the central cluster
            if self.num_clusters == 7:
                x_shift = np.array([7*d/2, -d/2, -4*d, -7*d/2, d/2, 4*d])
                y_shift = np.array([9*h, 15*h, 6*h, -9*h, -15*h, -6*h])
                for xs, ys in zip(x_shift, y_shift):
                    self.x = np.concatenate((self.x, x_central + xs))
                    self.y = np.concatenate((self.y, y_central + ys))

            self.x = np.repeat(self.x, self.num_sectors)
            self.y = np.repeat(self.y, self.num_sectors)

            if self.num_sectors == 3 and self.beamwidth < 0 :
                self.azimuth = np.tile(self.AZIMUTH, 19*self.num_clusters)
                self.elevation = np.tile(self.ELEVATION, 3*19*self.num_clusters)
            elif self.beamwidth < 0:
                self.azimuth = np.zeros(19*self.num_clusters)
                self.elevation = np.tile(-90, 19 * self.num_clusters)
            else:
                # calculate beams
                max_angle = np.arctan(self.intersite_distance / 2 / self.height) - np.deg2rad(self.beamwidth/2)
                max_dist = self.height * np.tan(max_angle)
                self.azimuth = np.zeros(self.num_sectors*19*self.num_clusters)
                self.elevation = np.tile(-90, 19 * self.num_sectors* self.num_clusters)

                self.beam_azimuth = np.empty(self.num_sectors*19*self.num_clusters )
                self.beam_elevation = np.empty(self.num_sectors * 19 * self.num_clusters)

                for cell_idx in range(19 * self.num_clusters):
                    done = False
                    while not done:
                        done = True

                        beam_x = random_number_gen.rand(self.num_sectors) * 2*max_dist - max_dist
                        beam_y = random_number_gen.rand(self.num_sectors) * 2*max_dist - max_dist

                        if any(beam_x**2 + beam_y**2 > max_dist ** 2):
                            done = False

                        for ii in range(self.num_sectors):
                            for jj in range(ii+1, self.num_sectors):
                                if (beam_x[ii] - beam_x[jj])**2 + (beam_y[ii] - beam_y[jj])**2 < self.beams_dist ** 2:
                                    done = False

                    azimuth_deg = np.rad2deg(np.arctan2(beam_y, beam_x))
                    elevation_deg = -np.rad2deg(np.arctan(self.height/np.sqrt(beam_x**2 + beam_y**2)))

                    self.beam_azimuth[cell_idx*self.num_sectors:(cell_idx+1)*self.num_sectors] = azimuth_deg
                    self.beam_elevation[cell_idx * self.num_sectors:(cell_idx + 1) * self.num_sectors] = elevation_deg

            # In the end, we have to update the number of base stations
            self.num_base_stations = len(self.x)

            self.indoor = np.zeros(self.num_base_stations, dtype = bool)

    def plot(self, ax: matplotlib.axes.Axes):

        # create the hexagons
        if self.num_sectors == 3:
            r = self.intersite_distance/3
            azimuth = self.azimuth
        else :
            r = self.intersite_distance/np.sqrt(3)
            azimuth = np.zeros(self.x.size)

        for x, y, az in zip(self.x, self.y, azimuth):
            if self.num_sectors == 3:
                angle = int(az - 60)
            else:
                x = x - self.intersite_distance/2
                y = y - r/2
                angle = int(az - 30)
            se = list([[x,y]])

            for a in range(6):
                se.extend([[se[-1][0] + r*math.cos(math.radians(angle)), se[-1][1] + r*math.sin(math.radians(angle))]])
                angle += 60
            sector = plt.Polygon(se, fill=None, edgecolor='k')
            ax.add_patch(sector)

        # macro cell base stations
        ax.scatter(self.x, self.y, color='k', edgecolor="k", linewidth=4, label="HAPS platforms")


if __name__ == '__main__':
    intersite_distance = 100
    num_clusters = 1
    num_sectors = 1

    topology = TopologyMacrocell(intersite_distance, num_clusters, num_sectors)
    topology.calculate_coordinates()

    fig = plt.figure(figsize=(8,8), facecolor='w', edgecolor='k')  # create a figure object
    ax = fig.add_subplot(1, 1, 1)  # create an axes object in the figure

    topology.plot(ax)

    plt.axis('image')
    plt.title("HAPS topology")
    plt.xlabel("x-coordinate [km]")
    plt.ylabel("y-coordinate [km]")
    plt.tight_layout()
    plt.show()


