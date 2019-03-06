#!/usr/bin/python
# -*- coding: utf-8 -*-
from os import walk
from os.path import join
import shutil 

mypath = "./"

f = open('all.txt','a')
for root, dirs, files in walk(mypath):
	for f1 in files:
		fullpath = join(root, f1)
		if 'Annotations' in fullpath:
			f.write(f1[:-4]+'\n')	




		
