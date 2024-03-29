# RISProcess
RISProcess is a Python package designed to download, process, and save seismic
data that were collected on the Ross Ice Shelf (RIS), Antarctica, from 2014-
2017. The package is principally built using [Obspy](https://docs.obspy.org)
and [h5py](https://www.h5py.org). The package is used to build the data set
required for the deep clustering project,
[RISCluster](https://github.com/NeptuneProjects/RISCluster). Details on the
clustering project are available in
[Jenkins et al.](https://doi.org/10.1029/2021JB021716).

Information about the seismic data set can be found at
[FDSN](https://www.fdsn.org/networks/detail/XH_2014/).  The project for which
the data were collected is documented in
[Bromirski et al.](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2015GL065284).

## Installation
Pre-requisites:
[Anaconda](https://anaconda.org) or
[Miniconda](https://docs.conda.io/en/latest/miniconda.html)

Tested on MacOS 11.1 and Red Hat Enterprise Linux 7.9.

The following steps will set up a Conda environment and install RISProcess.
1. Open a terminal and navigate to the directory you would like to download the
 **RISProcess.yml** environment file.
2. Save **RISProcess.yml** to your computer by running the following:
  <br>a. **Mac**:
  <br>`curl -LJO https://raw.githubusercontent.com/NeptuneProjects/RISProcess/master/RISProcess.yml`
  <br>b. **Linux**:
  <br>`wget --no-check-certificate --content-disposition https://raw.githubusercontent.com/NeptuneProjects/RISProcess/master/RISProcess.yml`
3. In terminal, run: `conda env create -f RISProcess.yml`
4. Once the environment is set up and the package is installed, activate your
environment by running `conda activate RISProcess` in terminal.

## Usage
The Jupyter notebook,
**[RISProcess.ipynb](https://github.com/NeptuneProjects/RISProcess/blob/master/RISProcess.ipynb)**,
provides an outline of general usage and workflow.  There are two components to
the worfklow: setting up configuration files, and executing scripts. The
configuration files can be set up manually (not recommended), or using the
provided notebook. Scripts are executed from the terminal, with recommended
commands printed within the notebook. Copy and paste the commands from the
notebook into terminal, taking care to ensure the working directories and path
names are consistent. Due to irregularities that can arise from executing
command line functions from within the iPython kernel, I chose to avoid calling
commands from within the notebook.

## Author
William Jenkins
<br>wjenkins [@] ucsd [dot] edu
<br>Scripps Institution of Oceanography
<br>University of California San Diego
<br>La Jolla, California, USA
