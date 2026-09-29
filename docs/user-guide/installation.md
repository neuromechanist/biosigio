# Installation

biosigIO uses [UV](https://docs.astral.sh/uv/) for Python environment and package
management. There are several ways to install biosigIO depending on your needs.

## From PyPI (Recommended)

The easiest way to install biosigIO is from PyPI:

```bash
uv pip install biosigio
```

If your own project is uv-managed, use `uv add biosigio` instead to track it as a dependency.

## From GitHub Repository

To install the latest development version directly from GitHub:

```bash
uv pip install "git+https://github.com/neuromechanist/biosigio.git"
```

Or clone and install:

```bash
git clone https://github.com/neuromechanist/biosigio.git
cd biosigio
uv pip install .
```

## Development Installation

For development purposes, install biosigIO in editable mode with dev dependencies:

```bash
git clone https://github.com/neuromechanist/biosigio.git
cd biosigio
uv sync --extra dev
```

This allows you to modify the source code and see the changes without reinstalling.

## Optional Dependencies

biosigIO provides optional dependency groups. For an existing install use
`uv pip install` (this is a UV-only project):

```bash
# Development tools (pytest, pytest-cov, ruff, ty)
uv pip install "biosigio[dev]"

# Documentation tools (mkdocs, mkdocs-material, mkdocstrings, mkdocs-jupyter, mike)
uv pip install "biosigio[docs]"

# MEG (.fif / CTF .ds / KIT .con,.sqd,.kdf / 4D-BTi) and BrainVision (.vhdr)
# import via MNE
uv pip install "biosigio[meg]"

# MEF3 iEEG (.mefd) import via MNE + pymef (needs a newer MNE than the `meg`
# extra alone; see the Data Formats > MEF3 page for the version requirement)
uv pip install "biosigio[mef3]"

# Parquet / Arrow / Feather serialization via pyarrow
uv pip install "biosigio[arrow]"

# Proprietary electrophysiology formats (Intan, Blackrock, Spike2, Plexon,
# Micromed, Neuralynx, ...) via python-neo
uv pip install "biosigio[neo]"

# Sharded Zarr v3 serving store via zarr v3
uv pip install "biosigio[zarr]"

# MATLAB v7.3 (HDF5) EEGLAB .set files via h5py (classic .set files need
# no extra)
uv pip install "biosigio[hdf5]"

# All optional dependencies
uv pip install "biosigio[all]"
```

## Dependencies

biosigIO has the following core dependencies:

| Dependency    | Purpose                | Minimum Version |
|---------------|------------------------|-----------------|
| numpy         | Array operations       | >=1.20.0        |
| pandas        | Data manipulation      | >=1.3.0         |
| scipy         | Signal processing      | >=1.7.0         |
| matplotlib    | Visualization          | >=3.4.0         |
| pyedflib      | EDF/BDF file handling  | >=0.1.30        |
| wfdb          | WFDB format support    | >=4.0.0         |
| pyxdf         | XDF/LSL format support | >=1.16.0        |

These dependencies are automatically installed when you install biosigIO.

## Python Version

biosigIO requires Python 3.11 or later. We test on Python 3.11, 3.12, 3.13, and 3.14.

## Testing

To run the tests:

```bash
uv sync --extra dev
uv run pytest
```

## Verifying Installation

Verify that biosigIO is correctly installed:

```python
import biosigio
print(biosigio.__version__)
```

You should see the version number (e.g., `1.2.9`) without any errors.
