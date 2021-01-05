# TMA severe weather alert data extraction

Scripts and functions for extracting weather alert data from Tanzanian
Meteorological Authority ``Five days Severe weather impact-based
forecasts'' PDFs.

These PDFs follow a standard format, with the first page showing
weather alerts for the day of issue and the second page describing
alerts for the subsequent four days. For each day, colour-coded
weather alert areas are drawn over a map of Tanzania, with weather
type indicated by symbols comprising a collection of vector graphics
and further details available in associated text.

This code seeks to extract weather alert information for the four days
subsequent to the issue date and to convert this information to
standardised gridded data that can be compared with observational
weather data.

## Requirements

The Python environment required by the code is specified in
`environment.yml`. Create the environment using
[conda](https://docs.conda.io/projects/conda/en/latest/index.html):

``` bash
conda env create -f environment.yml
```

The resulting environment is activated as necessary by the scripts or
it can be activated manually:

``` bash
conda activate tma_data_extraction
```

The code uses a third-party script `pdf-parser.py` to extract PDF
contents. Ensure that this script is available by placing it in the
root directory or by specifying its location in a `PDF_PARSER_ROOT`
environment variable:

``` bash
export PDF_PARSER_ROOT=path/to/pdf-parser.py
```

## Usage

To extract weather alert information from the documents in `data_dir`,
placing the results in `output_dir`, which is created if it does not
already exist:

``` bash
./extract_PDF_data.sh data_dir output_data_dir
```

The script produces netCDF files that contain separate data layers for
each alert area extracted from the PDF. Each layer is assigned an
alert level based on the colour of the alert area in the original
PDF. Currently, the weather type is not automatically assigned and
must be specified manually. Run the following script to validate the
netCDF data and assign weather type information:

``` bash
./check_alert_data.sh output_data_dir reviewed_data_dir
```

This script displays each alert area on a map of Tanzania and provides
options to adjust alert level and weather type and to flag areas for
review.
