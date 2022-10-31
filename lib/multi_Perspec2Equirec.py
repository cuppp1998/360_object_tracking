import os
import sys
import cv2
import numpy as np
import lib.Perspec2Equirec as P2E


class Perspective:
    def __init__(self, img_array, F_T_P_array, lon_maps_array, lat_maps_array):

        assert (
            len(img_array)
            == len(F_T_P_array)
            == len(lon_maps_array)
            == len(lat_maps_array)
        )

        self.img_array = img_array
        self.F_T_P_array = F_T_P_array
        self.lon_maps_array = lon_maps_array
        self.lat_maps_array = lat_maps_array

    def GetEquirec(self, height, width, original_height, original_weight):
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #
        merge_image = np.zeros((height, width, 3))
        merge_mask = np.zeros((height, width, 3))

        for img, [F, T, P], lon_map_original, lat_map_original in zip(
            self.img_array, self.F_T_P_array, self.lon_maps_array, self.lat_maps_array
        ):
            per = P2E.Perspective(
                img, F, T, P, lon_map_original, lat_map_original
            )  # Load equirectangular image
            img, mask = per.GetEquirec(
                height, width, original_height, original_weight
            )  # Specify parameters(FOV, theta, phi, height, width)
            merge_image += img
            merge_mask += mask

        merge_mask = np.where(merge_mask == 0, 1, merge_mask)
        merge_image = np.divide(merge_image, merge_mask)

        return merge_image
