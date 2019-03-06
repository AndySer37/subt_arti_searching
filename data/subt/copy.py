#!/usr/bin/python
# -*- coding: utf-8 -*-
from os import walk
from os.path import join
import shutil 
# Annotations JPEGImages

mypath = "./"
a = "./Annotations"
b = "./JPEGImages"

_cls = ["toolbox",'radio', 'backpack']
for root, dirs, files in walk(mypath):
	for f in files:
		fullpath = join(root, f)
		for i in range(len(_cls)):		
			if _cls[i] in fullpath:
				fn = _cls[i] + f[-10:]
		if 'annotation' in fullpath:
			shutil.copy(fullpath, join(a,fn))
		if 'image' in fullpath:
			shutil.copy(fullpath, join(b,fn))
		print (fullpath)


		
