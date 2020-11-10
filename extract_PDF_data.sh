#!/bin/bash

E_NOARGS=65
E_NODIR=67

if [ -z "$1" ]
then
  echo "Usage: `basename $0` data_dir out_dir"
  exit $E_NOARGS
else
    if [ -d "$1" ]
    then
        data_dir="${1%/}/"
    fi
fi

if [ -n "$2" ]
then
    out_dir="${2%/}/"
    if [ ! -d "$2" ]
    then
        # Create output directory
        mkdir -p $out_dir
    fi
else
    echo "Output directory not specified"
    echo "Usage: $0 data_dir out_dir"
    exit $E_NODIR
fi

if [ -z ${PDF_PARSER_ROOT:+x} ]
then
    # If not set or set but NULL, assume script is in current directory
    export PDF_PARSER_ROOT=$(getcwd)
fi

# Set up temporary data for intermediate files and specify log file
TEMP_DIR=${out_dir}temp
mkdir -p $TEMP_DIR
log_file=${out_dir}tma_extract.log

# Parse PDFs to get page contents
bash script/parse.sh $data_dir $TEMP_DIR > $log_file

# Extract weather alert metadata
Rscript R/pdf_extract.R $data_dir $TEMP_DIR >> $log_file
rm -rf $TEMP_DIR/*_raw_data.csv

# Process page contents and metadata and produce netcdf of weather alert data
for fil in $TEMP_DIR/*.csv
do
    base=$(basename -s '.csv' $fil)
    python python/process_content.py $TEMP_DIR/${base}_page2.txt $TEMP_DIR/${base}.csv >> $log_file
    mv *.nc $out_dir
done

# Clean up
rm -rf $TEMP_DIR
