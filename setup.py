from setuptools import setup, find_packages

setup(
    name="mp",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',
    ],
) 