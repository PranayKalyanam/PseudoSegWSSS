from __future__ import annotations

from data.patch.patch_coordinate import PatchCoordinate


class CoordinateTransform:
    """
    Utility class for converting coordinates between
    working-image space and original-image space.
    """

    @staticmethod
    def working_to_original(
        coordinate: PatchCoordinate,
        scale_factor: float,
    ) -> PatchCoordinate:
        """
        Convert working coordinates to original coordinates.
        """

        inv = 1.0 / scale_factor

        return PatchCoordinate(
            x=int(round(coordinate.x * inv)),
            y=int(round(coordinate.y * inv)),
            width=int(round(coordinate.width * inv)),
            height=int(round(coordinate.height * inv)),
        )

    @staticmethod
    def original_to_working(
        coordinate: PatchCoordinate,
        scale_factor: float,
    ) -> PatchCoordinate:
        """
        Convert original coordinates to working coordinates.
        """

        return PatchCoordinate(
            x=int(round(coordinate.x * scale_factor)),
            y=int(round(coordinate.y * scale_factor)),
            width=int(round(coordinate.width * scale_factor)),
            height=int(round(coordinate.height * scale_factor)),
        )

    @staticmethod
    def local_to_global(
        local_x: int,
        local_y: int,
        region_x: int,
        region_y: int,
    ) -> tuple[int, int]:
        """
        Convert region-local coordinates to global
        working-image coordinates.
        """

        return (
            region_x + local_x,
            region_y + local_y,
        )

    @staticmethod
    def global_to_local(
        global_x: int,
        global_y: int,
        region_x: int,
        region_y: int,
    ) -> tuple[int, int]:
        """
        Convert global coordinates to region-local coordinates.
        """

        return (
            global_x - region_x,
            global_y - region_y,
        )