from setuptools import setup

def setup_package():
    setup(
        name="RISProcess",
        url="https://github.com/NeptuneProjects/RISProcess.git",
        author="William F. Jenkins II",
        author_email="wjenkins@ucsd.edu",
        packages=["RISCluster"],
        package_dir = {"": "RISCluster"},
        install_requires=[
            "h5py",
            "jupyterlab",
            "numpy",
            "obspy",
            "pandas",
            "scipy",
            "tqdm"
        ],
        version="0.0b",
        license="MIT",
        description="Package provides signal processing workflow for seismic data \
            collected on the Ross Ice Shelf, Antarctica."
    )


if __name__ == "__main__":
    setup_package()
