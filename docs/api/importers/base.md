# Base Importer

The `BaseImporter` class is the abstract base class for all importers in biosigIO. It defines the interface that all specific importers must implement.

## Class Documentation

::: biosigio.importers.base
    options:
      show_root_heading: true
      show_source: true
      members: true

## Creating a Custom Importer

To create a custom importer for a new format, you should inherit from `BaseImporter` and implement its abstract methods:

```python
import numpy as np

from biosigio.core.emg import Recording
from biosigio.importers.base import BaseImporter


class MyCustomImporter(BaseImporter):
    def load(self, filepath: str, **kwargs) -> Recording:
        """
        Load and parse the custom format file.

        Args:
            filepath: Path to the input file.

        Returns:
            Recording: A Recording object with the loaded signals, channels,
            events, and metadata.
        """
        # Implement file loading logic for your custom format
        rec = Recording()

        # Add each channel (data + per-channel metadata)
        rec.add_channel(
            label='CH1',
            data=np.array([...]),       # Channel samples
            sample_frequency=1000,
            physical_dimension='µV',
            channel_type='EMG',
        )
        # ...additional channels...

        # Attach recording-level metadata
        rec.set_metadata('subject', '001')
        rec.set_metadata('recording_date', '2023-01-01')

        return rec
```

## Return Value

The `load()` method takes a `filepath` (and optional importer-specific keyword
arguments) and must return a `Recording` object. Populate it through the public
API rather than returning raw containers:

- **signals**: built up by `add_channel()`, which stores each channel as a
  column in `Recording.signals` (a pandas DataFrame indexed by time). Labels
  must be unique: `add_channel()` raises `ValueError` on a label the Recording
  already has, so an importer for a format that allows repeated labels has to
  make them unique first (the EDF, WFDB and Zarr importers use
  `biosigio.importers._labels.unique_channel_labels`).
- **channels**: each `add_channel()` call records per-channel metadata in
  `Recording.channels`:

  ```python
  {
      'channel_name': {
          'channel_type': str,        # Type of channel (e.g., 'EMG', 'ACC')
          'physical_dimension': str,  # Unit (e.g., 'µV', 'mV', 'g')
          'sample_frequency': float,  # Sampling frequency in Hz
          'modality': str,            # Inferred from channel_type if not given
          'prefilter': str,
      },
      # ...more channels...
  }
  ```

- **metadata**: recording-level fields set via `set_metadata()`:

  ```python
  {
      'subject': str,            # Subject identifier
      'recording_date': str,     # Recording date
      'device': str,             # Recording device
      # ...other metadata fields...
  }
  ```

## Using a Custom Importer

A custom importer is invoked directly by instantiating it (no constructor
arguments) and calling `load()` with the file path:

```python
from my_module import MyCustomImporter

rec = MyCustomImporter().load('data.custom')
```