import os
import sys
import cv2
import numpy as np


class Perspective:
    def __init__(self, img, FOV, THETA, PHI, lon_map_original, lat_map_original):
        self._img = img
        [self._height, self._width, _] = self._img.shape
        self.wFOV = FOV
        self.THETA = THETA
        self.PHI = PHI
        self.hFOV = float(self._height) / self._width * FOV

        self.w_len = np.tan(np.radians(self.wFOV / 2.0))
        self.h_len = np.tan(np.radians(self.hFOV / 2.0))

        self.lon_map_original = lon_map_original
        self.lat_map_original = lat_map_original

    def GetEquirec(self, height, width, original_height, original_width):
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #

        lon_map_original = self.lon_map_original / original_width * width
        lat_map_original = self.lat_map_original / original_height * height
        x, y = np.meshgrid(np.linspace(-180, 180, width), np.linspace(90, -90, height))

        x_map = np.cos(np.radians(x)) * np.cos(np.radians(y))
        y_map = np.sin(np.radians(x)) * np.cos(np.radians(y))
        z_map = np.sin(np.radians(y))

        xyz = np.stack((x_map, y_map, z_map), axis=2)

        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], np.float32)
        [R1, _] = cv2.Rodrigues(z_axis * np.radians(self.THETA))
        [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-self.PHI))

        R1 = np.linalg.inv(R1)
        R2 = np.linalg.inv(R2)

        xyz = xyz.reshape([height * width, 3]).T
        xyz = np.dot(R2, xyz)
        xyz = np.dot(R1, xyz).T

        xyz = xyz.reshape([height, width, 3])
        inverse_mask = np.where(xyz[:, :, 0] > 0, 1, 0)

        xyz[:, :] = xyz[:, :] / np.repeat(xyz[:, :, 0][:, :, np.newaxis], 3, axis=2)

        lon_map = np.where(
            (-self.w_len < xyz[:, :, 1])
            & (xyz[:, :, 1] < self.w_len)
            & (-self.h_len < xyz[:, :, 2])
            & (xyz[:, :, 2] < self.h_len),
            (xyz[:, :, 1] + self.w_len) / 2 / self.w_len * self._width,
            0,
        )
        lat_map = np.where(
            (-self.w_len < xyz[:, :, 1])
            & (xyz[:, :, 1] < self.w_len)
            & (-self.h_len < xyz[:, :, 2])
            & (xyz[:, :, 2] < self.h_len),
            (-xyz[:, :, 2] + self.h_len) / 2 / self.h_len * self._height,
            0,
        )
        mask = np.where(
            (-self.w_len < xyz[:, :, 1])
            & (xyz[:, :, 1] < self.w_len)
            & (-self.h_len < xyz[:, :, 2])
            & (xyz[:, :, 2] < self.h_len),
            1,
            0,
        )

        persp = cv2.remap(
            # self._img,
            cv2.rectangle(self._img, (300, 400), (800, 700), (255, 0, 255), 3),
            lon_map.astype(np.float32),
            lat_map.astype(np.float32),
            cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_WRAP,
        )

        cv2.imwrite("a.png", persp)
        print(lon_map.shape)

        original = cv2.rectangle(self._img, (300, 400), (800, 700), (255, 0, 255), 3)
        cv2.imwrite("or.png", original)
        xs = []
        ys = []
        for i in range(400, 700, 10):
            x = int(round(lon_map_original[i, 300]))
            y = int(round(lat_map_original[i, 300]))
            xs.append(x)
            ys.append(y)
            x = int(round(lon_map_original[i, 800]))
            y = int(round(lat_map_original[i, 800]))
            xs.append(x)
            ys.append(y)
        for i in range(300, 800, 10):
            x = int(round(lon_map_original[400, i]))
            y = int(round(lat_map_original[400, i]))
            xs.append(x)
            ys.append(y)
            x = int(round(lon_map_original[700, i]))
            y = int(round(lat_map_original[700, i]))
            xs.append(x)
            ys.append(y)

        a1 = min(xs)
        a2 = min(ys)
        b1 = max(xs)
        b2 = max(ys)
        print(a1, a2, b1, b2)
        # a1 = int(round(lon_map_original[400, 400]))
        # a2 = int(round(lat_map_original[400, 400]))
        # b1 = int(round(lon_map_original[700, 800]))
        # b2 = int(round(lat_map_original[700, 800]))
        # c1 = int(round(lon_map_original[700, 400]))
        # c2 = int(round(lat_map_original[700, 400]))
        # d1 = int(round(lon_map_original[400, 800]))
        # d2 = int(round(lat_map_original[400, 800]))
        # print(a1, a2, b1, b2)
        mask = mask * inverse_mask
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        persp = persp * mask

        cv2.imwrite("b.png", persp)
        img = cv2.imread("b.png", 3)
        after = cv2.rectangle(img, (a1, a2), (b1, b2), (255, 0, 255), 3)
        # after = cv2.circle(after, (c1, c2), 10, (255, 0, 255), 0)
        # after = cv2.circle(after, (d1, d2), 10, (255, 0, 255), 0)

        cv2.imwrite("after.png", after)

        # original = cv2.rectangle(self._img, (a1, a2), (b1, b2), (255, 0, 255), 3)
        # cv2.imwrite("or.png", original)
        return after, mask
