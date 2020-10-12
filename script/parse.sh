#!/bin/bash

E_NOARGS=65
E_NODIR=67

if [ -z "$1" ]
then
  echo "Usage: `basename $0` file"
  exit $E_NOARGS
fi

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

if [ -z ${PDF_PARSER_ROOT:+x} ]
then
    # If not set or set but NULL, assume script is in current directory
    PDF_PARSER=./pdf-parser.py
else
    PDF_PARSER=${PDF_PARSER_ROOT}/pdf-parser.py
fi

if [ -d "$1" ]
then
    in_dir="${1%/}/"
    for fil in "${in_dir}"*
    do
        bash script/parse.sh "${fil}" $out_dir
    done
    exit 0
fi

# Get cleaned up filename
filename=$(echo $1 | rev | cut -d'/' -f-1 | rev | cut -d'.' -f1)
filename=${filename// /_}
echo "Processing file: $filename"

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
    echo "Writing page data to: $outfile"
    python ${PDF_PARSER} -o $objnum -f -d $outfile "$1" >/dev/null
    let pnum=pnum+1
done
