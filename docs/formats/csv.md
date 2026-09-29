# Generic CSV Format

biosigIO provides a flexible CSV importer that can handle a wide variety of CSV-formatted data files, including those with or without headers, different delimiters, and various time column formats.

## Features

- Automatic detection of file format, headers, and delimiters
- Support for files with or without time columns
- Customizable channel properties
- Intelligent format detection to recommend specialized importers when appropriate

## Usage

```python
from biosigio import Recording

# Basic usage (automatic format detection)
rec = Recording.from_file('data.csv')

# Force using the generic CSV importer even if specialized format is detected
rec = Recording.from_file('data.csv', importer='csv', force_csv=True)

# Specify custom parameters
rec = Recording.from_file('data.csv', importer='csv', 
                   sample_frequency=1000,  # Required if no time column
                   has_header=True,        # Whether file has header row
                   delimiter=',',          # Column delimiter
                   skiprows=0,             # Rows to skip at the beginning
                   time_column='Time')     # Column to use as time index
```

## Specialized Parameters

The CSV importer accepts many customization parameters:

| Parameter | Description |
|-----------|-------------|
| `sample_frequency` | Sampling frequency in Hz (required if no time column) |
| `has_header` | Whether file has a header row (auto-detected if not specified) |
| `skiprows` | Number of rows to skip at beginning (auto-detected if not specified) |
| `delimiter` | Column delimiter (auto-detected if not specified) |
| `time_column` | Name or index of column to use as time index (auto-detected if not specified) |
| `columns` | List of column names or indices to include |
| `channel_names` | Custom names for channels |
| `channel_types` | Dict mapping column names to channel types ('EMG', 'ACC', etc.) |
| `physical_dimensions` | Dict mapping column names to physical dimensions |
| `metadata` | Dict of additional metadata to include |

## Examples

### Basic CSV with Headers

```python
# data.csv:
# Time,EMG1,EMG2,ACC1
# 0.000,0.1,0.2,0.3
# 0.001,0.2,0.3,0.4
# ...

rec = Recording.from_file('data.csv', importer='csv')
```

### Headerless CSV with Custom Names

```python
# data.csv:
# 0.1,0.2,0.3
# 0.2,0.3,0.4
# ...

rec = Recording.from_file('data.csv', importer='csv',
                    has_header=False,
                    sample_frequency=1000,  # Required since no time column
                    channel_names=['EMG_L', 'EMG_R', 'ACC_X'])
```

### Setting Channel Types and Units

```python
rec = Recording.from_file('data.csv', importer='csv',
                    channel_types={
                        'EMG1': 'EMG',
                        'EMG2': 'EMG',
                        'ACC1': 'ACC'
                    },
                    physical_dimensions={
                        'EMG1': 'mV',
                        'EMG2': 'mV',
                        'ACC1': 'g'
                    })
```

### Selecting Specific Columns

```python
# Only load columns 0 and 2
rec = Recording.from_file('data.csv', importer='csv',
                    columns=[0, 2],
                    sample_frequency=1000)

# Only load columns named 'EMG1' and 'ACC1'
rec = Recording.from_file('data.csv', importer='csv',
                    columns=['EMG1', 'ACC1'])
```

## Auto-Detection Features

The CSV importer includes several auto-detection capabilities:

- **Format Detection**: Recognizes specialized formats like Trigno CSV files
- **Delimiter Detection**: Identifies the most common delimiter (comma, tab, semicolon)
- **Header Detection**: Determines if the first row is a header based on content
- **Time Column Detection**: Looks for columns that might represent time

## Notes

- When a specialized format (like Trigno) is detected, the importer will suggest using the appropriate specialized importer
- For headerless files, auto-generated channel names will be "Channel_0", "Channel_1", etc.
- A header that repeats a column name keeps every column: pandas renames the repeats (`EMG`, `EMG.1`, ...). A repeated name in `channel_names` or `columns` raises `ValueError` naming it
- If no time column is found, a sample frequency must be provided to generate a time index 