#!/usr/bin/env python3

'''Database I/O functions.

William Jenkins, wjenkins [at] ucsd [dot] edu
Scripps Institution of Oceanography, UC San Diego
December 2020
'''
import configparser
import json
import os

import h5py
import numpy as np
import pandas as pd


def config_processing(parameters):
    config = configparser.ConfigParser()
    config["PARAMETERS"] = parameters
    fname = f"{parameters['path']}/config_{parameters['mode']}.ini"
    with open(fname, "w") as configfile:
        config.write(configfile)
    return fname


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
    '''Defines the structure of the .h5 database; h5py package required.
    Of note, pay special attention to the dimensions of the chunked data. By
    anticipating the output data dimensions, one can chunk the saved data on
    disk accordingly, making the reading process go much more quickly.'''
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
