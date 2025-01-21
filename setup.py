from setuptools import find_packages, setup

NAME = "shadertoy"
SUMMARY = "shadertoy"

VERSION = "0.0.1"

runtime_deps = ["wgpu>=0.19.0,<0.20.0"]

setup(
    name=NAME,
    version=VERSION,
    classifiers = [
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Visualization"
    ],
    packages=find_packages(
        include=["shadertoy", 'shadertoy.*'], exclude=["*.tests", "*.tests.*", "tests.*", "tests"]
    ),
    python_requires=">=3.9.0",
    install_requires=runtime_deps,
    license="MIT",
    description=SUMMARY,
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Pan Xinmiao",
    author_email="pan_xinmiao@163.com",
    url="https://github.com/panxinmiao/shadertoy",
    data_files=[("", ["LICENSE"])]
)
