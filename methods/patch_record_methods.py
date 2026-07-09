from methods.patch_label_methods import PatchLabelMethods


class PatchRecordMethods:

    @staticmethod
    def label_string(record):

        return "".join(
            map(
                str,
                record.labels,
            )
        )

    @staticmethod
    def has_positive_label(record):

        return any(
            record.labels
        )

    @staticmethod
    def coordinate(record):

        return (
            record.x,
            record.y,
        )