# RISProcess
RISProcess is a Python package designed to download, process, and save seismic
data that were collected on the Ross Ice Shelf (RIS), Antarctica, from 2014-
2017.  The package is built using [Obspy](https://docs.obspy.org), h5py, and a
number of other libraries.

This package is used to build the data sets required for the deep embedded
clustering project, [RISCluster](https://github.com/NeptuneProjects/RISCluster).  
Details on the clustering project are available in [Jenkins et al.](https://www.essoar.org/doi/abs/10.1002/essoar.10505894.1)

Information on the data set can be found at [FDSN](https://www.fdsn.org/networks/detail/XH_2014/). The project is documented in [Bromirski et al.](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1002/2015GL065284)

## Installation
### Mac & Linux
Pre-requisites:
[Anaconda](https://anaconda.org) or
[Miniconda](https://docs.conda.io/en/latest/miniconda.html)

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

## Author
William Jenkins
Scripps Institution of Oceanography
UC San Diego
La Jolla, California
