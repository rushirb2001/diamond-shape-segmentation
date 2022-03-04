"""
Setup configuration for diamond-shape-segmentation package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="diamond-shape-segmentation",
    version="1.0.0",
    author="Rushir Bhavsar, Harshil Sanghvi, Ruju Shah, Vrunda Shah, Khushi Patel",
    author_email="rushirbhavsar@gmail.com",
    description="Computer Vision pipeline for automated diamond shape segmentation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rushirbhavsar/diamond-shape-segmentation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
)