from setuptools import setup, find_packages

with open("README.md") as f:
    long_description = f.read()

setup(
    name="mlp-mixer",
    packages=find_packages(),
    version="0.0.1",
    license="MIT",
    description="Pytorch implementation of MLPMixer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Himanshu Dutta",
    author_email="meet.himanshu.dutta@gmail.com",
    url="https://github.com/himanshu-dutta/MLPMixer-pytorch",
    keywords=["deep learning", "image classification"],
    install_requires=["torch>=1.6"],
    python_requires=">3.6",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
