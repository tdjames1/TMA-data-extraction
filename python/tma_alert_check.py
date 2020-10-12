#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tma_alert_check
.. module:: TMA-data-extraction
    :synopis: Scripts and functions for extracting weather alert data
    from Tanzanian Meteorological Authority "Five days Severe weather
    impact-based forecasts" PDFs.
.. moduleauthor: Tamora D. James <t.d.james1@leeds.ac.uk>, CEMAC (UoL)
.. description: This module was developed by CEMAC as part of the GCRF
African Swift Project. This script allow manual checking of alerts
generated by the PDF processing routines.
   :copyright: © 2020 University of Leeds.
   :license: BSD 3-clause (see LICENSE)
Example:
    To use::
        ./tma_alert_check.py <nc_dir>
        <nc_dir> - Directory containing netcdf files to be checked
.. CEMAC_cemac_generic:
   https://github.com/cemac/cemac_generic
"""

import sys
import argparse
import fnmatch
from os import listdir, mkdir
from shutil import copyfile
from datetime import datetime
from warnings import simplefilter, catch_warnings

import netCDF4
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_alert(lon, lat, dat, extent = None, title = None):

    cmap = ListedColormap([(1,1,1,0), "yellow", "orange", "red"])

    if extent is None:
        extent = [min(lon), max(lon), min(lat), max(lat)]

    fig, ax = plt.subplots(subplot_kw = {
        'projection': ccrs.PlateCarree()
    })
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS)

    ax.contourf(lon, lat, dat, transform=ccrs.PlateCarree(),
                cmap=cmap, vmin=0, vmax=3)

    if title is not None:
        plt.title(title)

    plt.show(block=False)
    # end plot_alert()

def check_value(prompt, vals, retries=4, reminder='Please try again!'):
    while True:
        val = input(prompt)
        if val in vals:
            return val
        retries = retries - 1
        if retries < 0:
            raise ValueError('invalid user response')
        print(reminder)
    # end check_value()

def check_alert_level(nc, var):

    data = nc[var][:]
    level = np.amax(data)
    level_s = "No warning"
    if level == 1.0:
        level_s = "Advisory"
    elif level == 2.0:
        level_s = "Warning"
    elif level == 3.0:
        level_s = "Major warning"

    msg = "Alert level is {:d} ({:s}). Accept (y/n)? ".format(int(level), level_s)
    opt = ('y', 'n')
    accept_level = check_value(msg, opt)
    if accept_level == "n":
        msg = "Select alert level: 1 (Advisory)/2 (Warning)/3 (Major warning): "
        opt =  ('1', '2', '3')
        new_level = check_value(msg, opt)
        data[data == level] = float(new_level)
        nc[var][:] = data
        level = new_level
    nc[var].setncattr('alert_level', int(level))
    # end check_alert_level()

def check_alert_type(nc, var):
    alert_type = nc[var].getncattr('alert_type')
    msg = "Alert type is {:s}. Accept (y/n)? ".format(alert_type)
    opt = ('y', 'n')
    accept_type = check_value(msg, opt)
    if accept_type == "n":
        msg = (
            "Alert type is {:s}. Select type: w(ind)/r(ain)/t(idal)/o(ther): "
            .format(alert_type)
        )
        opt = ('w','r','t','o')
        alert_type = check_value(msg, opt)
        if alert_type == 'w':
            alert_type = 'Strong wind'
        elif alert_type == 'r':
            alert_type = 'Heavy rain'
        elif alert_type == 't':
            alert_type = 'Tidal/waves'
        elif alert_type == 'o':
            alert_type = input("Enter alert type: ")
        nc[var].setncattr('alert_type', alert_type)
    # end check_alert_type()

def check_alert_flag(nc, var):

    msg = "Flag alert for review (y/n)? "
    opt = ('y', 'n')
    flag_alert = check_value(msg, opt)

    is_flagged = False
    try:
        flag_text = nc[var].getncattr('alert_flag')
    except AttributeError:
        pass
    else:
        is_flagged = True

    if flag_alert == "y":
        if is_flagged:
            msg = "Current flag text: {:s}. Replace (y/n)? ".format(flag_text)
            opt = ('y', 'n')
            replace_text = check_value(msg, opt)
        if is_flagged and replace_text == "y" or not is_flagged:
            flag_text = input("Enter text: ")
            nc[var].setncattr('alert_flag', flag_text)
    elif is_flagged:
        msg = "Current flag text: {:s}. Remove (y/n)? ".format(flag_text)
        opt = ('y', 'n')
        remove_text = check_value(msg, opt)
        if (remove_text == "y"):
            nc[var].delncattr('alert_flag')
    # end check_alert_flag()

# Main
def main():

    parser = argparse.ArgumentParser(description='Check TMA alert data')
    parser.add_argument('input_dir', nargs=1, type=str)
    parser.add_argument('output_dir', nargs=1, type=str)
    args = parser.parse_args()

    input_dir = args.input_dir[0]
    files = [f for f in listdir(input_dir) if fnmatch.fnmatch(f, '*.nc')]

    output_dir = args.output_dir[0]
    try:
        mkdir(output_dir)
    except FileExistsError:
        print("Using existing output directory")

    in_fmt = "%Y-%m-%d"
    out_fmt = "%d-%m-%Y"

    for f in files[0:1]:
        src = "/".join((input_dir, f))
        dst = "/".join((output_dir, f))
        copyfile(src, dst)

        nc = netCDF4.Dataset(dst, mode='r+', format='NETCDF4')
        lon = nc.variables['lon'][:]
        lat = nc.variables['lat'][:]
        extent = [min(lon), max(lon), min(lat), max(lat)]

        issue_dt = datetime.strptime(getattr(nc, 'issue_date'), in_fmt)
        issue_date = issue_dt.strftime(out_fmt)

        alert_vars = [v for v in nc.variables.keys() if "alert" in v]

        for v in alert_vars:
            var = nc[v]
            alert_data = var[:][:]

            alert_level = np.amax(alert_data)

            masked_data = np.ma.array(alert_data, mask=alert_data<alert_level)

            alert_date = datetime.strptime(var.getncattr('alert_date'), in_fmt).strftime(out_fmt)
            alert_id = var.getncattr('alert_id')
            title = "Alert #{:d}, {:s}".format(alert_id, alert_date)

            with catch_warnings():
                simplefilter("ignore")
                plot_alert(lon, lat, masked_data, extent, title)
                plt.pause(0.0001)

            print("")
            print("### Reviewing alert data for TMA warning issued on", issue_date, "###")
            print("")
            print("Alert date:", alert_date)
            print("Alert id:", alert_id)
            print("Alert text:", var.getncattr('alert_text'), "\n")

            check_alert_level(nc, v)
            check_alert_type(nc, v)
            check_alert_flag(nc, v)

            plt.close()
            print("Please wait for the next figure to be plotted...\n")

        # write to file
        nc.close()

        msg = "Finished reviewing warnings for {:"+out_fmt+"}. Continue (y/)n? "
        opt = ('y', 'n')
        ctn = check_value(msg.format(issue_dt), opt)
        if ctn == "n":
            break

    # end of main()

# Run main
if __name__ == "__main__":
    main()
