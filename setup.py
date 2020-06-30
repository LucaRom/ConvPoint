from setuptools import setup, find_packages

setup(
    name='ConvPoint',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        "laspy","plyfile"
    ],
    url='https://github.com/sfoucher/ConvPoint',
    license='',
    author='fouchesa',
    author_email='',
    description='Modifications of convpoint'
)
