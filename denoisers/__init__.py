"""Dataset-specific comparison denoisers and their protocol identifiers."""

N2N_IMPLEMENTATION = "neighbor2neighbor_official_v1"


def check_n2n_checkpoint(config):
    """Prevent the previous custom objective's weights from becoming an official N2N run."""
    if config.get("treatment") == "n2n" and config.get("n2n_implementation") != N2N_IMPLEMENTATION:
        raise ValueError(
            "This N2N checkpoint predates the official implementation or uses a different protocol. "
            "Refit N2N with the current code. Historical captured results belong to the legacy project baseline."
        )
