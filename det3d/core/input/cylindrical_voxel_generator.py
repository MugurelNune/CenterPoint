import numpy as np
from det3d.ops.point_cloud.point_cloud_ops import points_to_cylindrical_voxel


class CylindricalVoxelGenerator:
    def __init__(self, fixed_volume_space, point_cloud_range, grid_size):
        self._point_cloud_range = point_cloud_range
        self._fixed_volume_space = fixed_volume_space
        self._max_volume_space = point_cloud_range[3:]
        self._min_volume_space = point_cloud_range[:3]
        self._grid_size = grid_size

    def generate(self, points):
        return points_to_cylindrical_voxel(
            points,
            self._fixed_volume_space,
            self._max_volume_space,
            self._min_volume_space,
            self._grid_size,
        )

    @property
    def fixed_volume_space(self):
        return self._fixed_volume_space

    @property
    def max_volume_space(self):
        return np.asarray(self._max_volume_space)

    @property
    def min_volume_space(self):
        return np.asarray(self._min_volume_space)

    @property
    def grid_size(self):
        return np.asarray(self._grid_size)

    @property
    def point_cloud_range(self):
        return np.asarray(self._point_cloud_range)
