#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import ctypes
import warnings

# CuDNN settings
_available = True
_version_list = [4, 5, 6]
if sys.platform in ("linux", "linux2"):
    _libcudnn_libname_list = ['libcudnn.so'] + \
        ['libcudnn.so.%s' % v for v in _version_list]
elif sys.platform == "darwin":
    _libcudnn_libname_list = ["libcudnn.dylib"] +\
        ["libcudnn.%s.dylib" % v for v in _version_list]
elif sys.platform == "win32":
    _libcudnn_libname_list = ["libcudnn64_%s.dll" % v for v in _version_list]
else:
    _available = False

_libcudnn = None

for _libcudnn_libname in _libcudnn_libname_list:
    try:
        _libcudnn = ctypes.cdll.LoadLibrary(_libcudnn_libname)
    except OSError:
        pass
    else:
        break

if _libcudnn is None:
    warnings.warn("Could not find cudnn libraries.")
