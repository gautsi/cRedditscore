#!/usr/bin/env python
# -*- coding: utf-8 -*-


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


with open('README.rst') as readme_file:
    readme = readme_file.read()

requirements = [
    'praw==3.1.0',
    'SQLAlchemy==1.0.8',
    'pandas==0.16.2'
]

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='cRedditscore',
    version='0.1.0',
    description="An automatic comment flagger for online forums",
    long_description=readme,
    author="Gautam Sisodia",
    author_email='gautam.sisodia@gmail.com',
    url='https://github.com/gautsi/cRedditscore',
    packages=[
        'cRedditscore',
    ],
    package_dir={'cRedditscore':
                 'cRedditscore'},
    include_package_data=True,
    install_requires=requirements,
    license="BSD",
    zip_safe=False,
    keywords='cRedditscore',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
