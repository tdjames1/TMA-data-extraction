#!/usr/bin/env Rscript
args = commandArgs(trailingOnly=TRUE)
if (length(args) == 0) {
    stop("At least one argument must be supplied (input dir).n", call.=FALSE)
} else if (length(args) == 1) {
    ## default output dir
    args[2] = "out/"
}

source("R/extract_data.R")

files <- list.files(args[1], "*.pdf$", full.names = TRUE)

purrr::walk(files, extract_data, out_dir = args[2])
