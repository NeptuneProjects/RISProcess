"""This script provides to read raw MSEED seismic data into processed MSEED
data.  Functions available allow for filtering, decimation, and instrument
response removal.  Single- and multi-core processing are available.  If the
script is in debug mode, parameters are set within the script.  Otherewise,
parameters are passed via the command line execution of the script.

Example
-------
In Terminal:
python process.py 20141202T0000 20161129T0000 --sourcepath '/Volumes/RISData'
--writepath '/Users/williamjenkins/Research/Data/MSEED' --network 'XH'
--taper 60 --prefeed 60 --fs2 50 --cutoff 3 20.1 --output 'ACC'
--prefilt 0.004 0.01 500 1000 --waterlevel 14 --num_workers 12

William Jenkins, wjenkins [at] ucsd [dot] edu
Scripps Institution of Oceanography, UC San Diego
December 2020
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime
import importlib as imp

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

import core
import workflows
imp.reload(core)
imp.reload(workflows)


debug = False
if __name__ == "__main__":
    tic = datetime.now()
    if debug:
        start = "20141202T000000"
        stop = "20141203T000000"
        sourcepath = "/Volumes/RISData"
        catalogue = "/Users/williamjenkins/Research/Data/MSEED/" + \
            "catalogue_cleaned.csv"
        writepath = "/Users/williamjenkins/Desktop/RISData.h5"
        network = "XH"
        station = "RS17"
        channel = "HHZ"
        params = core.SignalProcessing(
            start,
            stop,
            mode="cat2h5",
            sourcepath=sourcepath,
            writepath=writepath,
            catalogue=catalogue,
            network=network,
            station=station,
            channel=channel,
            taper=10,
            prefeed=10,
            fs2=50,
            cutoff=(3,20),
            T_seg=4,
            NFFT=256,
            tpersnap=0.4,
            overlap=0.9,
            output="acc",
            prefilt=(0.004, 0.01, (50/2)*20, (50/2)*40),
            waterlevel=14,
            detector="z",
            STA=3,
            LTA=60,
            on=8,
            off=4,
            det_window=5,
            num_workers=1
        )
    else:
        parser = argparse.ArgumentParser(
            description="Pre-processes data from MSEED."
        )
        parser.add_argument(
            "start",
            help="Enter start date/time YYYYMMDDTHHMMSS."
        )
        parser.add_argument(
            "stop",
            help="Enter stop date/time YYYYMMDDTHHMMSS."
        )
        parser.add_argument(
            "mode",
            help="Processing mode [ process | detect ]"
        )
        parser.add_argument(
            "--sourcepath",
            default=".",
            help="Enter path to data archive."
        )
        parser.add_argument(
            "--catalogue",
            default=".",
            help="Enter path to event catalogue."
        )
        parser.add_argument(
            "--writepath",
            default=".",
            help="Enter path to save."
        )
        parser.add_argument("--name_format", default=1, type=int)
        parser.add_argument("--network", default="*")
        parser.add_argument("--station", default="*")
        parser.add_argument("--location", default="*")
        parser.add_argument("--channel", default="*")
        parser.add_argument("--taper", type=float)
        parser.add_argument("--prefeed", type=float)
        parser.add_argument("--fs2", type=float)
        parser.add_argument("--cutoff", nargs="+", type=float)
        parser.add_argument("--T_seg", type=float)
        parser.add_argument("--NFFT", type=int)
        parser.add_argument("--tpersnap", type=float)
        parser.add_argument("--overlap", type=float)
        parser.add_argument("--output")
        parser.add_argument("--prefilt", nargs="+", type=float)
        parser.add_argument("--waterlevel", type=float)
        parser.add_argument("--detector")
        parser.add_argument("--STA", type=float)
        parser.add_argument("--LTA", type=float)
        parser.add_argument("--on", type=float)
        parser.add_argument("--off", type=float)
        parser.add_argument("--det_window", type=float)
        parser.add_argument("--num_workers", type=int, default=1)
        parser.add_argument("--verbose", type=int, default=0)
        args = parser.parse_args()
        params = core.SignalProcessing(**vars(args))
    print("=" * 79)
    print(f"Processing data.  Workers: {params.num_workers}")
    start_search = params.start.floor('D')
    stop_search = params.stop.floor('D')
    dts = pd.date_range(start_search, stop_search)[:-1]
    A = list()
    for dt in dts:
        params_copy = deepcopy(params)
        params_copy.update_times(dt, dt + pd.Timedelta(1, "D"))
        A.append(dict(params=params_copy))
    count = 0
    pbargs = {
        'total': int(len(dts)),
        'unit_scale': True,
        'bar_format': '{l_bar}{bar:20}{r_bar}{bar:-20b}',
        'leave': True
    }
    if (params.mode == "preprocess") or (params.mode == "detect"):
        if params.num_workers == 1:
            for a in tqdm(A, **pbargs):
                count += workflows.process_data(**a)
        else:
            with ProcessPoolExecutor(max_workers=params.num_workers) as exec:
                futures = [exec.submit(workflows.process_data, **a) for a in A]
                for future in tqdm(as_completed(futures), **pbargs):
                    count += future.result()

        print(f"Processing complete.")
        if params.mode == "preprocess":
            print(f"{count} files saved to {params.writepath}")
        elif params.mode == "detect":
            print(f"{count} detections; catalogue saved to {params.writepath}")
    elif params.mode == "cat2h5":
        core.init_h5datasets(params)
        if params.num_workers == 1:
            for a in tqdm(A, **pbargs):
                count += workflows.build_h5(**a)
        else:
            with ProcessPoolExecutor(max_workers=params.num_workers) as exec:
                MAX_JOBS_IN_QUEUE = params.num_workers
                jobs = {}
                days_left = len(A)
                days_iter = iter(A)

                while days_left:
                    for a in days_iter:
                        job = exec.submit(workflows.build_h5, **a)
                        jobs[job] = a
                        if len(jobs) > MAX_JOBS_IN_QUEUE:
                            break
                    for job in as_completed(jobs):
                        days_left -= 1
                        count += job.result()
                        a = jobs[job]
                        del jobs[job]
                        print(
                            f"{len(dts) - days_left}/{len(dts)} days ",
                            f"({100*(len(dts) - days_left)/len(dts):.1f}%) ",
                            f"processed.",
                            flush=True,
                            end="\r"
                        )
                        break
        print(f"\n{count} samples saved.")
    time_elapsed = str(datetime.now() - tic)[:-7]
    print(f"{time_elapsed} elapsed.")
    print("=" * 79)
