from setuptools import setup, find_packages

setup(
    name='neural_control',
    version='0.0.1',
    install_requires=['torch', 'numpy', 'matplotlib', 'scipy', 'pyglet', 'ruamel.yaml', 'tqdm'],
    packages=find_packages()
)
