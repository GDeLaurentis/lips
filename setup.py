from setuptools import setup, find_packages

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='lips',
    version='v0.3.0',
    license='GNU General Public License v3.0',
    description='Lorentz Invariant Phase Space',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Giuseppe De Laurentis',
    author_email='g.dl@hotmail.it',
    url='https://github.com/GDeLaurentis/lips',
    download_url='https://github.com/GDeLaurentis/lips/archive/v0.3.0.tar.gz',
    project_urls={
        'Documentation': 'https://gdelaurentis.github.io/lips/',
        'Issues': 'https://github.com/GDeLaurentis/lips/issues',
    },
    keywords=['lips', 'Lorentz Invariant Phase Space', 'Spinor Helicity'],
    packages=find_packages(),
    install_requires=['numpy',
                      'mpmath',
                      'sympy', ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 2.7',
    ],
)
