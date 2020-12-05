# distutils: language = c++
# distutils: sources = Wrapper.cpp

# Cython interface file for wrapping the object
#
#

from libcpp.vector cimport vector

# c++ interface to cython
cdef extern from "Wrapper.h" namespace "pclwrapper":
  cdef cppclass Wrapper:
        Wrapper() except +
        vector[long double] compute(vector[vector[double ]])

# creating a cython wrapper class
cdef class PyPCL:
    cdef Wrapper *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self):
        self.thisptr = new Wrapper()
    def __dealloc__(self):
        del self.thisptr

    def compute(self, sv):
        return self.thisptr.compute(sv)