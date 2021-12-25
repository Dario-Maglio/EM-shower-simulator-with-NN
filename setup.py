"""Install the package in Development mode writing from bash the command
   pip install -e . It allows to use the simulate_shower module in python.
"""

from EM_shower_simulator import PACKAGE_NAME, AUTHOR, AUTHOR_EMAIL, DESCRIPTION, URL

from setuptools import setup, find_packages

import versioneer



with open("README.md", "r") as f:
    _LONG_DESCRIPTION = f.read()
with open("LICENSE", "r") as f:
    _LICENSE = f.readline().strip()
with open('requirements.txt', 'r') as f:
    _DEPENDENCIES = f.read().splitlines()

_CLASSIFIERS = [
    'License :: OSI Approved :: '
    'GNU General Public License v3',
    'Operating System :: OS Independent',
    "Programming Language :: Python :: 3",
    'Programming Language :: C++',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific computation',
    'Development Status :: Beta']
_SCRIPTS = [
    'scripts/simulate_EM_shower'
]
_PACKAGES = find_packages(exclude='tests')

_KWARGS = dict(name=PACKAGE_NAME,
               version=versioneer.get_version(),
               cmdclass=versioneer.get_cmdclass(),
               author=AUTHOR,
               author_email=AUTHOR_EMAIL,
               description=DESCRIPTION,
               long_description=_LONG_DESCRIPTION,
               long_description_content_type="text/markdown",
               license=_LICENSE,
               url=URL,
               classifiers=_CLASSIFIERS,
               python_requires='>=3.7',
               install_requires=_DEPENDENCIES,
               scripts=_SCRIPTS,
               packages=_PACKAGES,
               include_package_data=True)


setup(**_KWARGS)
