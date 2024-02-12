import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

VERSION = "0.1.0"
PACKAGE_NAME = "hython"
AUTHOR = "Mohammad Alasawedah"
AUTHOR_EMAIL = "masawdah@gmail.com"
URL = "https://gitlab.inf.unibz.it/REMSEN/intertwin-hython"

LICENSE = "Apache License 2.0"
DESCRIPTION = "A python package aims to exploit state-of-the-art timeseries forcasting for hydrologic models."
LONG_DESCRIPTION = (HERE / "README.md").read_text()
LONG_DESC_TYPE = "text/markdown"

INSTALL_REQUIRES = ["numpy", "pandas","geopandas","cartopy", "xarray", "dask" ]

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESC_TYPE,
    author=AUTHOR,
    license=LICENSE,
    author_email=AUTHOR_EMAIL,
    url=URL,
    python_requires='>=3.9',
    install_requires=INSTALL_REQUIRES,
    packages=find_packages(),
)
