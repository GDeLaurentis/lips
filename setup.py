from setuptools import setup, find_packages


setup(
    name='lips',
    version='0.0.1',
    author='Giuseppe De Laurentis',
    author_email='g.dl@hotmail.it',
    description='Lorentz Invariant Phase Space',
    packages=find_packages(),
    install_requires=['numpy==1.14.2',
                      'mpmath==1.0.0', ],
)