"""Version information for biosigIO."""

__version__ = "1.2.8"
__version_info__ = (1, 2, 8)


def get_version() -> str:
    """Get the current version string."""
    return __version__


def get_version_info() -> tuple:
    """Get the version info tuple (major, minor, patch)."""
    return __version_info__
