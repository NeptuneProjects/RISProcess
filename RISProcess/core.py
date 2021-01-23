'''Seismic signal processing classes and functions.

William Jenkins, wjenkins [at] ucsd [dot] edu
Scripps Institution of Oceanography, UC San Diego
December 2020
'''
from datetime import datetime
import json
import os

import h5py
import numpy as np
from obspy import read, read_inventory
from obspy.core import UTCDateTime
import pandas as pd
from scipy import signal


class SignalProcessing():
    def __init__(
            self,
            start,
            stop,
            mode,
            sourcepath=".",
            name_format=1,
            writepath=None,
            catalogue=None,
            network="*",
            station="*",
            location="*",
            channel="*",
            taper=None,
            prefeed=None,
            fs2=None,
            cutoff=None,
            T_seg=None,
            NFFT=None,
            tpersnap=None,
            overlap=None,
            output=None,
            prefilt=None,
            waterlevel=None,
            detector=None,
            STA=None,
            LTA=None,
            on=None,
            off=None,
            det_window=None,
            num_workers=1,
            verbose=0
        ):
        self.mode = mode
        self.sourcepath = sourcepath
        if writepath is not None:
            self.writepath = writepath
        else:
            self.writepath = f"./ProcessedData"
        self.catalogue = catalogue
        self.name_format = name_format
        self.network = network
        self.station = station
        self.channel = channel
        self.taper = taper
        self.prefeed = prefeed
        self.fs2 = fs2
        self.cutoff = cutoff
        self.T_seg = T_seg
        self.NFFT = NFFT
        self.tpersnap = tpersnap
        self.overlap = overlap
        self.output = output
        if prefilt is not None:
            if isinstance(prefilt, list):
                prefilt = tuple(prefilt)
        self.prefilt = prefilt
        self.waterlevel = waterlevel
        self.detector = detector
        self.STA = STA
        self.LTA = LTA
        self.on = on
        self.off = off
        self.det_window = det_window
        self.num_workers = num_workers
        if verbose == 0:
            self.verbose = False
        elif (verbose == 1) and (num_workers == 1):
            self.verbose = True
        self.update_times(start, stop)


    def update_times(self, start, stop):
        '''Updates time specifications.

        Parameters
        ----------
        start : str
            Start date-time

        stop : str
            Stop date-time

        Returns
        -------
        None

        Notes
        -----
        The intended use case of this function is to allow for iterative
        updates to the dates/times of interest, without having to specify the
        signal processing parameters with each iteration.
        '''
        self.start = pd.Timestamp(start)
        self.stop = pd.Timestamp(stop)

        if (self.taper is not None) and (self.prefeed is not None):
            self.buffer_front = self.taper + self.prefeed
            self.buffer_back = self.taper
        elif self.taper is not None:
            self.buffer_front = self.taper
            self.buffer_back = self.taper
        elif self.prefeed is not None:
            self.buffer_front = self.prefeed
            self.buffer_back = 0
        else:
            self.buffer_front = 0
            self.buffer_back = 0

        self.start_processing = self.start - pd.Timedelta(seconds=self.buffer_front)
        self.stop_processing = self.stop + pd.Timedelta(seconds=self.buffer_back)


def centered_spectrogram(tr, params):
    fs = tr.stats.sampling_rate
    npersnap = fs * params.tpersnap
    window = np.kaiser(npersnap, beta=5.7)
    f, t, S = signal.spectrogram(
        tr.data,
        fs,
        window,
        nperseg=npersnap,
        noverlap=params.overlap*npersnap,
        nfft=params.NFFT
    )
    dtvec = params.start_processing + pd.to_timedelta(t, "sec")
    tmask = (dtvec >= params.start) & (dtvec < params.stop)
    dtvec_ = dtvec[tmask]
    S_sum = np.sum(S, axis=0)
    pk_idx = np.argmax(S_sum[tmask])
    pk_dt = dtvec_[pk_idx]
    tmask_c = (dtvec >= pk_dt - pd.Timedelta(params.T_seg/2, "sec")) & (dtvec < pk_dt + pd.Timedelta(params.T_seg/2, "sec"))
    fmask = (f >= params.cutoff[0]) & (f <= params.cutoff[1])
    t = t[tmask_c]
    t = t - min(t)
    f = f[fmask]
    S = S[:,tmask_c][fmask,:]
    tvec = np.insert(t, 0, np.NAN)
    S_out = np.hstack((f[:,None], S))
    S_out = np.vstack((S_out, tvec))
    return t, f, S, S_out, dtvec[tmask_c][0], dtvec[tmask_c][-1]


def clean_catalogue(
        source,
        dest=f"{pd.Timestamp.now().strftime('%y%m%d%H%M%S')}.csv",
        window=None
    ):
    '''Removes duplicate entries in detection catalogue and sorts events by
    datetime.

    Parameters
    ----------
    source : str
        Path to catalogue (.csv)
    dest : str
        Path to destination file (.csv)
    window : float
        Window (s) from previous detection in which subsequent detections will
        be removed.
    '''
    catalogue = pd.read_csv(source, parse_dates=[3,4,5])
    catalogue.drop_duplicates(ignore_index=True, inplace=True)
    catalogue.sort_values(by=["dt_on"], ignore_index=True, inplace=True)
    if window is not None: # 430367
        count = 0
        rm_idx = []
        for station in catalogue.station.unique():
            subset = catalogue.loc[catalogue["station"] == station]
            remove = []
            dt_on_ = subset["dt_on"].iloc[0]
            for i, dt_on in enumerate(subset["dt_on"]):
                if i == 0:
                    continue
                elif dt_on < dt_on_ + pd.Timedelta(window, unit="sec"):
                    remove.append(i)
                else:
                    dt_on_ = dt_on
            rm_idx.append(subset.index[remove])
            count += len(remove)
        print(f"Removing {count} entries...")
        rm_idx = tuple([item for sub_idx in rm_idx for item in sub_idx])
        catalogue.drop(catalogue.index[[rm_idx]], inplace=True)
        catalogue.reset_index(drop=True, inplace=True)
    catalogue.to_csv(dest)
    print(f"Catalogue saved to {dest}.")


def clean_detections(npts, on_off):
    '''Removes spurious seismic detections that occur within a window following
    a detection.

    Parameters
    ----------
    npts : int
        Length of window in data samples.
    on_off : array
        On/off indexes for detections.

    Returns
    -------
    array
        Corrected on/off indexes.
    '''
    on = on_off[:,0]
    off = on_off[:,1]
    idx_on = [on[0]]
    idx_off = [off[0]]
    lowest_idx = on[0]

    for ion, ioff in zip(on, off):
        if ion > lowest_idx + npts:
            idx_on.append(ion)
            idx_off.append(ioff)
            lowest_idx = ion

    return np.asarray((idx_on, idx_off)).T


def collect_results(future, params):
    output = future.result()
    if any(map(lambda x: x is None, output)):
        print("No detections found.")
        return
    else:
        tr = output[0]
        S = output[1]
        C = output[2]
        metadata = output[3]
        print("Writing results to h5...")
        write_h5datasets(tr, S, C, metadata)


def decimate_to_fs2(st, fs2):
    '''Decimates traces in stream to a common sampling rate.

    Parameters
    ----------
    st : Stream
        Obspy object containing seismic traces read from disk.
    fs2 : int
        Sampling rate to which the traces are resampled.  The ratio between
        the original sampling rate and fs2 must be an integer.

    Returns
    -------
    st : Stream
        Obspy object containing seismic traces read from disk.
    '''
    if fs2 is not None:
        for tr in st:
            fs = tr.stats.sampling_rate
            factor = int(round(fs/fs2))
            tr.decimate(factor=factor)
    return st


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


def pipeline(params):
    if params.verbose:
        print("Reading stream.")
    st = read_stream(params)
    if st == -1:
        if params.verbose:
            print("No files found.")
        return 0
    if params.verbose:
        print("Removing gap traces.")
    st = remove_gap_traces(st)
    if params.verbose:
        print("Detrending.")
    st.detrend(type="Linear")
    if params.taper is not None:
        if params.verbose:
            print("Tapering.")
        st.taper(max_percentage=0.5, type='hann', max_length=params.taper)
    if params.fs2 is not None:
        if params.verbose:
            print("Decimating.")
        st = decimate_to_fs2(st, params.fs2)
    if (params.waterlevel and params.output and params.prefilt) is not None:
        if params.verbose:
            print("Reading station XML.")
        inv = read_stationXML(params.sourcepath, params.network, params.station)
        if params.verbose:
            print("Removing response.")
        st.remove_response(inventory=inv, water_level=params.waterlevel, output=params.output, pre_filt=params.prefilt, plot=False)
    if params.cutoff is not None:
        if params.verbose:
            print("Filtering.")
        st.filter('bandpass', freqmin=params.cutoff[0], freqmax=params.cutoff[1], zerophase=True)
    return st


def read_stationXML(sourcepath, network, station):
    '''Reads seismic station XML file.

    Parameters
    ----------
    sourcepath : str
        Path to station XML file
    network : str
        Seismic network
    station : str
        Seismic station

    Returns
    -------
    inv : Inventory
        Obspy Inventory object; contains instrument response information.
    '''
    filespec = f"{network}.{station}.xml"
    inv = read_inventory(f"{sourcepath}/StationXML/{filespec}")
    return inv


def read_stream(params):
    '''Reads MSEED data from file according to search parameters specified in
    params.

    Parameters
    ----------
    params : SignalProcessing object
        Object containing signal processing parameters passed.

    Returns
    -------
    st : Stream
        Obspy object containing seismic traces read from disk.
    -1 : int
        Returned if no files were read.
    '''
    start_search = params.start_processing.floor('D')
    stop_search = params.stop_processing.floor('D')
    dts = pd.date_range(start_search, stop_search)
    count = 0
    for i, dt in enumerate(dts):
        if params.name_format == 1:
            filespec = f"{params.network}.{params.station}.{params.channel}.{dt.year}.{dt.dayofyear:03d}.mseed"
        elif params.name_format == 2:
            filespec = f"{params.network}.{params.station}..{params.channel}__{dt.year}{dt.month:02d}{dt.day:02d}T*"

        if count == 0:
            try:
                st = read(f"{params.sourcepath}/MSEED/{filespec}")
                count += 1
            except:
                pass
        else:
            try:
                st += read(f"{params.sourcepath}/MSEED/{filespec}")
                count += 1
            except:
                pass
    if count > 0:
        st.merge()
        st.trim(
            starttime=UTCDateTime(params.start_processing),
            endtime=UTCDateTime(params.stop_processing)
        )
        return st
    else:
        return -1


def remove_gap_traces(st):
    '''Searches for gaps in seismic data and removes affected traces from
    stream.

    Parameters
    ----------
    st : Stream
        Obspy object containing seismic traces read from disk.

    Returns
    -------
    st : Stream
        Obspy object containing seismic traces read from disk.

    Notes
    -----
    Without this function in the workflow, interpolation/gap-filling methods
    can cause discontinuities in the data, which leads to delta functions
    during filtering and instrument response removal.
    '''
    # Check 1:
    gaps = st.get_gaps()
    for i in range(len(gaps)):
        for tr in st.select(network=gaps[i][0], station=gaps[i][1], location=gaps[i][2], channel=gaps[i][3]):
            st.remove(tr)
    # Check 2:
    masked = [np.ma.is_masked(tr.data) for tr in st]
    for i, tr in enumerate(st):
        if masked[i]:
            st.remove(tr)
    return st


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
