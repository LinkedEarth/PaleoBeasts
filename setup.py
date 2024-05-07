from setuptools import setup, find_packages

version = '0.1.0'

setup(
    name='paleobeasts',
    version=version,
    author='Alex James, Jordan Landers, Julien Emile-Geay',
    author_email='akjames@usc.edu',
    package_dir={"": "."},
    packages=find_packages(),
    description='A package for generating synthetic paleoclimate data',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    license='MIT',
    url='http://github.com/LinkedEarth/PaleoBeasts',
    download_url='https://github.com/LinkedEarth/PaleoBeasts/tarball/'+version,
)