try:
    # noinspection PyUnresolvedReferences
    import RPi
    is_raspberry = True
except (ImportError, RuntimeError):
    is_raspberry = False
