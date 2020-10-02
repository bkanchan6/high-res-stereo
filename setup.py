from setuptools import setup, find_packages
import os

version = "0.0.1"
if "VERSION" in os.environ:
    version = os.environ["VERSION"]


setup(
    name="high-res-stereo",
    version=version,
    description="high-res-stereo",
    author="Jariullah Safi",
    author_email="safijari@isu.edu",
    packages=find_packages(),
    install_requires=["torch", "opencv-python"],
)
