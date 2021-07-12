#!/usr/bin/env python3

"""Seismic signal processing classes and functions.

William Jenkins, wjenkins [at] ucsd [dot] edu
Scripps Institution of Oceanography, UC San Diego
January 2021
"""
from datetime import datetime
import json
import warnings

import numpy as np
from obspy import read, read_inventory
from obspy.core import UTCDateTime
from obspy.io.mseed.headers import InternalMSEEDWarning
import pandas as pd
from scipy import signal

from RISProcess.io import write_h5datasets


class SignalProcessing():
    def __init__(
            self,
            start,
            stop,
            mode,
            sourcepath='.',
            name_format=1,
            writepath='./ProcessedData',
            catalogue='.',
            parampath='.',
            network='*',
            station='*',
            location='*',
            channel='*',
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
        self.writepath = writepath
        self.catalogue = catalogue
        self.parampath = parampath
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
        """Updates time specifications.

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
        """
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


    def save_json(self, path=None):
        """Saves class keys and values to JSON file.

        Parameters
        ----------
        path : str
            Path to save JSON file.
        """
        if path is None:
            path = self.parampath

        params = {str(key): str(value) for key, value in self.__dict__.items()}
        with open(f'{path}/params_{self.mode}.json', 'w') as f:
            json.dump(params, f)


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
    """Removes duplicate entries in detection catalogue and sorts events by
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
    """
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
        rm_idx = [item for sub_idx in rm_idx for item in sub_idx]
        catalogue.drop(catalogue.index[rm_idx], inplace=True)
        catalogue.reset_index(drop=True, inplace=True)
    catalogue.to_csv(dest)
    print(f"Catalogue saved to {dest}.")


def clean_detections(npts, on_off):
    """Removes spurious seismic detections that occur within a window following
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
    """
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
    """Decimates traces in stream to a common sampling rate.

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
    """
    if fs2 is not None:
        for tr in st:
            fs = tr.stats.sampling_rate
            factor = int(round(fs/fs2))
            tr.decimate(factor=factor)
    return st


def pipeline(params):
    # Reads data from file
    if params.verbose:
        print("Reading stream.")
    st = read_stream(params)
    if st == -1:
        if params.verbose:
            print("No files found.")
        return 0
    # If only reading stream, function returns stream here.
    if params.mode == "detect":
        return st
    # Remove any traces with gaps
    if params.verbose:
        print("Removing gap traces.")
    st = remove_gap_traces(st)
    # Detrend data
    if params.verbose:
        print("Detrending.")
    st.detrend(type="polynomial", order=3)
    # Taper data
    if params.taper is not None:
        if params.verbose:
            print("Tapering.")
        st.taper(max_percentage=0.5, type='hann', max_length=params.taper)
    # Remove instrument response
    if (params.waterlevel and params.output and params.prefilt) is not None:
        if params.verbose:
            print("Reading station XML.")
        inv = read_stationXML(params.sourcepath, params.network, params.station)
        if params.verbose:
            print("Removing response.")
        st.remove_response(inventory=inv, water_level=params.waterlevel, output=params.output, pre_filt=params.prefilt, plot=False)
    # Decimate data
    if params.fs2 is not None:
        if params.verbose:
            print("Decimating.")
        st = decimate_to_fs2(st, params.fs2)
    # Filter data
    if params.cutoff is not None:
        if params.verbose:
            print("Filtering.")
        st.filter('bandpass', freqmin=params.cutoff[0], freqmax=params.cutoff[1], zerophase=True)
    return st


def read_stationXML(sourcepath, network, station):
    """Reads seismic station XML file.

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
    """
    filespec = f"{network}.{station}.xml"
    inv = read_inventory(f"{sourcepath}/StationXML/{filespec}")
    return inv


def read_stream(params):
    """Reads MSEED data from file according to search parameters specified in
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
    """
    # Ignore file integrity issues; thus far the only station affected is DR11,
    # with no seeming impact on the seismic trace itself. Consider treating as
    # an error in future implementation.
    warnings.simplefilter("error", category=InternalMSEEDWarning)

    start_search = params.start_processing.floor('D')
    stop_search = params.stop_processing.floor('D')
    dts = pd.date_range(start_search, stop_search)
    count = 0
    for i, dt in enumerate(dts):
        if params.name_format == 1:
            filespec = f"{params.network}/{params.station}/{params.network}.{params.station}.{params.channel}.{dt.year}.{dt.dayofyear:03d}.mseed"
        elif params.name_format == 2:
            filespec = f"{params.network}.{params.station}..{params.channel}__{dt.year}{dt.month:02d}{dt.day:02d}T*"
        print(filespec)
        # if count == 0:
        #     try:
        #         st = read(f"{params.sourcepath}/MSEED/{filespec}")
        #         count += 1
        #     except:
        #         pass
        # else:
        #     try:
        #         st += read(f"{params.sourcepath}/MSEED/{filespec}")
        #         count += 1
        #     except:
        #         pass

        # try:
        if count == 0:
            st = read(f"{params.sourcepath}/MSEED/{filespec}")
        else:
            st += read(f"{params.sourcepath}/MSEED/{filespec}")
        count += 1
        # except:
            # pass

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
    """Searches for gaps in seismic data and removes affected traces from
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
    """
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
