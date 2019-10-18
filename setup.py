from setuptools import setup, find_packages


setup(
    name='lips',
    version='v0.1.3',
    license='GNU General Public License v3.0',
    description='Lorentz Invariant Phase Space',
    long_description='Documentation at https://gdelaurentis.github.io/lips/',
    author='Giuseppe De Laurentis',
    author_email='g.dl@hotmail.it',
    url='https://github.com/GDeLaurentis/lips',
    download_url='https://github.com/GDeLaurentis/lips/archive/v0.1.3.tar.gz',
    keywords=['lips', 'Lorentz Invariant Phase Space', 'Spinor Helicity'],
    packages=find_packages(),
    install_requires=['numpy<1.17',
                      'mpmath<=1.1.0',
                      'sympy<=1.4', ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 2.7',
    ],
)
