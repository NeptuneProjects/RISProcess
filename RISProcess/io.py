#!/usr/bin/env python3

"""Database & workflow I/O functions.

William Jenkins, wjenkins [at] ucsd [dot] edu
Scripps Institution of Oceanography, UC San Diego
January 2021
"""
import configparser
import json
import logging
import os

import h5py
import numpy as np
from obspy import UTCDateTime
from obspy.clients.fdsn.mass_downloader import MassDownloader, \
    RectangularDomain, Restrictions
import pandas as pd


def config(mode, path=".", parameters=None):
    """Writes or reads configuration file for RISProcess.

    Parameters
    ----------
    mode : str
        Sets function to write mode ["w"] or read mode ["r"]

    path : str
        Sets path to write or read file (default: cwd).

    parameters : dict
        Dictionary containing configuration parameters to write; required for
        write mode (default: None).

    Returns
    -------
    fname : str
        In write mode, returns path to saved config file.

    parameters : dict
        In read mode, returns dictionary of formatted config parameters.
    """
    if mode == "w":
        if parameters is not None:
            config = configparser.ConfigParser()
            config.optionxform = str
            config["PARAMETERS"] = parameters
            # fname = f"{path}/config_{parameters['mode']}.ini"
            fname = os.path.join(path, f"config_{parameters['mode']}.ini")
            with open(fname, "w") as configfile:
                config.write(configfile)
            return fname
        else:
            raise TypeError("'parameters' required when in write mode.")
    elif mode == "r":
        dict_of_dtypes = {
            "name_format": "int",
            "taper": "float",
            "prefeed": "float",
            "fs2": "float",
            "cutoff": "arrayfloat",
            "T_seg": "float",
            "NFFT": "int",
            "tpersnap": "float",
            "overlap": "float",
            "prefilt": "arrayfloat",
            "waterlevel": "float",
            "STA": "float",
            "LTA": "float",
            "on": "float",
            "off": "float",
            "det_window": "float",
            "num_workers": "int",
            "verbose": "int"
        }
        config = configparser.ConfigParser()
        config.optionxform = str
        config.read(path)
        parameters = dict()
        for key, value in config["PARAMETERS"].items():
            if key in dict_of_dtypes.keys():
                if dict_of_dtypes[key] == 'float':
                    parameters[key] = float(config["PARAMETERS"][key])
                elif dict_of_dtypes[key] == 'int':
                    parameters[key] = int(config["PARAMETERS"][key])
                elif dict_of_dtypes[key] == 'arrayfloat':
                    parameters[key] = [float(i) for i in config['PARAMETERS'][key].split(', ')]
                elif dict_of_dtypes[key] == 'arrayfloat':
                    parameters[key] = [int(i) for i in config['PARAMETERS'][key].split(', ')]
            else:
                parameters[key] = value
        return parameters


def init_h5datasets(params):
    with h5py.File(params.writepath, 'a') as f:
        group_name = f"{params.T_seg:.1f}"
        if f"/{group_name}" not in f:
            print(f"No h5 group found, creating group '{group_name}' and datasets.")
            h5_name = f.create_group(group_name)
            h5_name.attrs["T_seg (s)"] = params.T_seg
            h5_name.attrs["NFFT"] = params.NFFT
            dset_tr, dset_spec, dset_scal, dset_cat = make_h5datasets(
                params.T_seg,
                params.NFFT,
                params.tpersnap,
                50,
                h5_name,
                params.overlap
            )


def make_h5datasets(T_seg, NFFT, tpersnap, fs, group_name, overlap):
    """Defines the structure of the .h5 database; h5py package required.
    Of note, pay special attention to the dimensions of the chunked data. By
    anticipating the output data dimensions, one can chunk the saved data on
    disk accordingly, making the reading process go much more quickly."""
    # Set up dataset for traces:
    m = 0
    n = 199
    dset_tr = group_name.create_dataset(
        'Trace',
        (m, n),
        maxshape=(None,n),
        chunks=(20, n),
        dtype='f'
    )
    dset_tr.attrs['AmplUnits'] = 'Velocity (m/s)'
    # Set up dataset for spectrograms:
    m = 0
    # n = int(NFFT/2 + 1) + 1
    n = 88
    o = 101
    dset_spec = group_name.create_dataset(
        'Spectrogram',
        (m, n, o),
        maxshape=(None, n, o),
        chunks=(20, n, o),
        dtype='f'
    )
    dset_spec.attrs['TimeUnits'] = 's'
    dset_spec.attrs['TimeVecXCoord'] = np.array([1,200])
    dset_spec.attrs['TimeVecYCoord'] = 70
    dset_spec.attrs['FreqUnits'] = 'Hz'
    dset_spec.attrs['FreqVecXCoord'] = 0
    dset_spec.attrs['FreqVecYCoord'] = np.array([0,68])
    dset_spec.attrs['AmplUnits'] = '(m/s)^2/Hz'
    # Set up dataset for scalograms:
    m = 0
    # n = int(NFFT/2 + 1)
    n = 69
    o = int(T_seg/(tpersnap*(1 - overlap)))
    dset_scal = group_name.create_dataset(
        'Scalogram',
        (m, n, o),
        maxshape=(None, n, o),
        chunks=(20, n, o),
        dtype='f'
    )

    # Set up dataset for catalogue:
    m = 0
    dtvl = h5py.string_dtype(encoding='utf-8')
    dset_cat = group_name.create_dataset(
        'Catalogue',
        (m,),
        maxshape=(None,),
        dtype=dtvl
    )
    return dset_tr, dset_spec, dset_scal, dset_cat


def write_h5datasets(tr, S, metadata, params):
    M = tr.shape[0]
    if not os.path.exists(f"{params.writepath}.csv"):
        pd.DataFrame.from_dict(metadata).to_csv(f"{params.writepath}.csv", mode="a", index=False)
    else:
        pd.DataFrame.from_dict(metadata).to_csv(f"{params.writepath}.csv", mode="a", index=False, header=False)
    with h5py.File(params.writepath, 'a') as f:
        group_name = f"{params.T_seg:.1f}"
        dset_tr = f[f'/{group_name}/Trace']
        dset_spec = f[f'/{group_name}/Spectrogram']
        # dset_scal = f[f'/{group_name}/Scalogram']
        dset_cat = f[f'/{group_name}/Catalogue']

        dset_tr.resize(dset_tr.shape[0]+M, axis=0)
        dset_spec.resize(dset_spec.shape[0]+M, axis=0)
        # dset_scal.resize(dset_scal.shape[0]+M, axis=0)
        dset_cat.resize(dset_cat.shape[0]+M, axis=0)

        dset_tr[-M:,:] = tr
        dset_spec[-M:,:,:] = S
        # dset_scal[-m:,:,:] = C
        for i in np.arange(0,M):
            dset_cat[-M+i,] = json.dumps(metadata[i])
    return M


def FDSN_downloader(
        datapath,
        start='20141201',
        stop='20161201',
        network='XH',
        station='*',
        channel='HH*',
        **kwargs
    ):
    """This function uses the FDSN mass data downloader to automatically
    download data from the XH network deployed on the RIS from Dec 2014 - Dec
    2016. More information on the Obspy mass downloader available at:
    https://docs.obspy.org/packages/autogen/obspy.clients.fdsn.mass_downloader.html

    Parameters
    ----------
    datapath : str
        Path to save MSEED and XML data.

    start : str
        Start date, in format YYYYMMDD

    stop : str
        Stop date, in format YYYYMMDD

    network : str
        2-character FDSN network code

    station: str
        2-character station code

    channel: str
        3-character channel code
    """
    print("=" * 65)
    print("Initiating mass download request.")
    start = UTCDateTime(start)
    stop  = UTCDateTime(stop)

    if not os.path.exists(datapath):
        # os.makedirs(f'{datapath}/MSEED')
        os.makedirs(os.path.join(datapath, 'MSEED'))
        # os.makedirs(f'{datapath}/StationXML')
        os.makedirs(os.path.join(datapath, 'StationXML'))

    domain = RectangularDomain(
        minlatitude=-85,
        maxlatitude=-75,
        minlongitude=160,
        maxlongitude=-130
    )

    restrictions = Restrictions(
        starttime = start,
        endtime = stop,
        chunklength_in_sec = 86400,
        network = network,
        station = station,
        location = "*",
        channel = channel,
        reject_channels_with_gaps = True,
        minimum_length = 0.0,
        minimum_interstation_distance_in_m = 100.0
    )

    mdl = MassDownloader(providers=["IRIS"])
    mdl.download(
        domain,
        restrictions,
        # mseed_storage=f"{datapath}/MSEED",
        mseed_storage=os.path.join(datapath, 'MSEED'),
        # stationxml_storage=f"{datapath}/StationXML"
        stationxml_storage=os.path.join(datapath, 'StationXML'),
    )

    logger = logging.getLogger("obspy.clients.fdsn.mass_downloader")
    logger.setLevel(logging.DEBUG)
