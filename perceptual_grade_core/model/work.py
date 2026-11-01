"""Cooperative progress ranges for pure numerical worker functions."""


def span(checkpoint, start, stop):
    if checkpoint is None:
        return None

    def check(progress=None):
        if progress is None:
            checkpoint()
        else:
            checkpoint(start+(stop-start)*progress)

    return check
