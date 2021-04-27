from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    "matplotlib==3.4.1",
    "numpy==1.19.2",
    "pandas",
    "tensorflow",
    "google-cloud-storage==1.37.1",
    "opencv-python",
    "SciPy"
]

setup(
    name="trainer",
    version="0.4",
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='dog-cat trainer'
)
