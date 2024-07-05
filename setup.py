from setuptools import setup

setup(
    name="stljax",
    version="0.0.1",
    description="stlcg with jax",
    author="Karen Leung",
    author_email="kymleung@uw.edu",
    packages=["stljax"],
    install_requires=[
        "jax",
        "matplotlib",
        "numpy"
        "graphviz"
    ],
)