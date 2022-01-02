import pathlib
from setuptools import find_packages, setup

HERE = pathlib.Path(__file__).parent

setup(
    name="nfractal",
    version="0.2.0",
    description="Create Fractals Using Complex-Valued Neural Networks!",
    long_description="Create Fractals Using Complex-Valued Neural Networks!\n Visit the homepage for more information.",
    long_description_content_type="text/markdown",
    url="https://github.com/amirabbasasadi/neural-fractal.git",
    author="Amirabbas Asadi",
    author_email="amir137825@gmail.com",
    packages=find_packages(exclude=("docs",)),
    include_package_data=True,
    install_requires=["tqdm", "torch>=1.8.1"],
)
