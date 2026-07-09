class PatchLabelMethods:

    @staticmethod
    def to_list(label):

        return [
            label.tumor,
            label.stroma,
            label.lymphocyte,
            label.necrosis,
        ]

    @staticmethod
    def to_string(label):

        return (
            f"{label.tumor}"
            f"{label.stroma}"
            f"{label.lymphocyte}"
            f"{label.necrosis}"
        )

    @staticmethod
    def has_positive(label):

        return any(
            PatchLabelMethods.to_list(label)
        )