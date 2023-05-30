from setuptools import setup, find_packages

setup(
    name='neural_control',
    version='0.0.1',
    install_requires=[
        'torch', 
        'gym',
        'numpy', 
        'matplotlib', 
        'scipy', 
        'pyglet', 
        'ruamel.yaml', 
        'tqdm', 
        'casadi', 
        'pandas',
        'scikit-learn',
        'pyquaternion',
        'tensorboard',
        ],
    packages=find_packages()
)
