from setuptools import setup, find_packages

setup(
    name="src",  # Changed from mapf_algorithms to src
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "networkx",
    ]
)