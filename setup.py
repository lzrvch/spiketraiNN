"""spiketrainn: a lightweight Python library for spike train distance calculation."""

import os
import re
import codecs
import setuptools

here = os.path.abspath(os.path.dirname(__file__))

with open('README.md', 'r') as fh:
    LONG_DESCRIPTION = fh.read()


def read(*parts):
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


DISTNAME = 'spiketrainn'
VERSION = '0.1'
DESCRIPTION = 'spiketrainn: a lightweight Python library for spike train distance calculation.'
MAINTAINER = 'Ivan Lazarevich'
MAINTAINER_EMAIL = 'ivan@lazarevi.ch'
URL = 'https://github.com/vanyalzr/spiketraiNN'
DOWNLOAD_URL = 'https://github.com/vanyalzr/spiketraiNN'
LONG_DESCRIPTION_CONTENT_TYPE = 'text/markdown'


INSTALL_REQUIRES = [
    'numpy==1.18.1',
    'scikit_learn==0.22.1',
    'Cython'
]

EXTRAS_REQUIRE = {'tests': ['pytest'], 'docs': []}

setuptools.setup(
    name=DISTNAME,
    version=VERSION,
    author=MAINTAINER,
    author_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
    url=URL,
    download_url=DOWNLOAD_URL,
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
)
