#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 10:50:27 2021

@author: hemerson
"""

from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'AgentRL'
LONG_DESCRIPTION = 'A package containing several lightweight reinforcement learning agents'

# TODO: update the install requirements

# Setting up
setup(
        name="AgentRL", 
        version=VERSION,
        author="Harry Emerson",
        author_email="emersonharry8@gmail.com",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[
            ],        
        keywords=['reinforcement learning', 'agent'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)
