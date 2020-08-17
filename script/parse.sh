#!/bin/bash

# Default location of pdf-parser.py
PDF_PARSER=pdf-parser.py

E_NOARGS=65
E_NODIR=67

if [ -z "$1" ]
then
  echo "Usage: `basename $0` file"
  exit $E_NOARGS
fi

# Get cleaned up filename
filename=$(echo $1 | rev | cut -d'/' -f-1 | rev | cut -d'.' -f1)
filename=${filename// /_}
echo "$filename"

out_dir=""
if [ -d "$2" ]
then
    out_dir="${2%/}/"
    echo "Using output directory: $out_dir"
else
    if [ -n "$2" ]
    then
        echo "$2 is not a directory"
        echo "Usage: $0 file [out_dir]"
        exit $E_NODIR
    fi
fi

get_pages() {
    python ${PDF_PARSER} -a "$1" | grep "\Page " | cut -d ':' -f 2 | cut -d',' -f1- --output-delimiter=''
}

# Get pages for parsing
pnum=1
for page_ref in $(get_pages "$1")
do
    # Get object number for page contents
    objnum=$(python ${PDF_PARSER} -o $page_ref "$1" | grep "/Contents" | cut -c15- | cut -d' ' -f1)
    echo "$objnum"
    outfile="${out_dir}${filename}_page${pnum}.txt"
    echo "$outfile"
    python ${PDF_PARSER} -o $objnum -f -d $outfile "$1" >/dev/null
    let pnum=pnum+1
done
