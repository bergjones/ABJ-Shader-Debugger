'''
/*
optimized-ggx.hlsl
This file describes several optimizations you can make when creating a GGX shader.
AUTHOR: John Hable
LICENSE:
This is free and unencumbered software released into the public domain.
Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.
In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
For more information, please refer to <http://unlicense.org/>
*/
'''
import bpy
import bmesh
import math
import mathutils
from datetime import datetime
from contextlib import contextmanager
import random
import numpy as np
import importlib

class myTest:
	def __init__(self):
		super(myTest, self).__init__()

		cube0 = 0

		# pass


	def G1V(self, dotNV, k):
		cube1 = 0 
		return 1.0 / (dotNV * (1.0 - k) + k)
	
	def testPrint(self):
		myString = '~~~~~~~~~~~ myString...from TestPrint in GGX_hable_abj 006'
		return myString