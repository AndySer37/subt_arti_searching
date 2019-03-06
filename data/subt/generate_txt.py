#!/usr/bin/python
# -*- coding: utf-8 -*-
from os import walk
from os.path import join
import shutil 

mypath = "./"

f1 = open('train.txt','a')
f2 = open('val.txt','a')
f3 = open('all.txt','a')
count = 0
for root, dirs, files in walk(mypath):
	
	for f in files:
		count += 1
		fullpath = join(root, f)
		if 'Annotations' in fullpath:
			if count % 10 != 0:
				f1.write(f[:-4]+'\n')	
			else:
				f2.write(f[:-4]+'\n')
			f3.write(f[:-4]+'\n')