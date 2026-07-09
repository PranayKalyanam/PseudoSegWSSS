class CoordinateMethods:

    @staticmethod
    def as_tuple(coordinate):

        return (
            coordinate.x,
            coordinate.y,
        )

    @staticmethod
    def shape(coordinate):

        return (
            coordinate.width,
            coordinate.height,
        )