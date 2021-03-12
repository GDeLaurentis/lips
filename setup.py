from setuptools import setup, find_packages


setup(
    name='lips',
    version='0.0.1',
    author='Giuseppe De Laurentis',
    author_email='g.dl@hotmail.it',
    description='Lorentz Invariant Phase Space',
    packages=find_packages(),
    include_package_data=True,
    # data_files=[('lips', ['lips/hardcoded_limits/DoubleCollinearLimit_D&S_a',
    #                       'lips/hardcoded_limits/DoubleCollinearLimit_D&S_e', ])],
    install_requires=['numpy',
                      'mpmath',
                      'sympy', ],
)
