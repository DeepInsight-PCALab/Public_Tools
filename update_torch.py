#!/usr/bin/env python
# coding=utf-8

cmd = '''conda install pytorch torchvision cuda80 -c soumith << EOF
Y
'''
import os
for x in range(10):
    print(cmd)
    os.system(cmd)

