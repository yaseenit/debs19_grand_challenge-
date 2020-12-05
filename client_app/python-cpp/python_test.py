# Python running example


import numpy as np
from my_pcl import PyPCL
import pcl
import timeit



# Initialising the wrapped c++ function
pyPCL = PyPCL();



cloud = pcl.load('/home/yassin/debs/objects/labeled_objects/Atm-673.pcd')

print(pyPCL.compute(np.array(cloud, dtype=np.float32)))

# def fun():
# 	pyPCL.compute(cloud.to_array())

# if __name__ == '__main__':
#     import timeit
#     print(timeit.timeit("fun()", setup="from __main__ import fun",number=1000))



  # start = time.time()
  # end = time.time()
  # print(end - start)
