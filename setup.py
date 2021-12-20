#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 10:50:27 2021

@author: hemerson
"""

from setuptools import setup, find_packages

VERSION = '0.0.9' 
DESCRIPTION = 'AgentRL'
LONG_DESCRIPTION = 'A package containing several lightweight reinforcement learning agents'

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
            'numpy==1.21.4',           
    	    'setuptools==58.0.4',
    	    'torch==1.10.0'            
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
