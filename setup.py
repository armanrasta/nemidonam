# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 14:11:31 2023

@author: AI
"""

from cx_Freeze import Executable, setup
import sys
sys.setrecursionlimit(100000)

setup(name="objectdetection",
      version="1.0",
      description="detect object for improve safety in industrial workplace",
      executables=[Executable("main.py")])
