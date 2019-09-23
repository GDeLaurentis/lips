from setuptools import setup, find_packages


setup(
    name='lips',
    version='0.0.1',
    author='Giuseppe De Laurentis',
    author_email='g.dl@hotmail.it',
    url='https://github.com/GDeLaurentis/lips-lite',
    description='Lorentz Invariant Phase Space',
    packages=find_packages(),
    install_requires=['numpy<1.17',
                      'mpmath<=1.1.0',
                      'sympy<=1.4', ],
)
