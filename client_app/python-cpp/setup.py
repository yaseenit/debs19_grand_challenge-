


from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext = Extension('my_pcl', sources=["my_pcl.pyx"], language="c++", extra_compile_args=['-std=c++11','-fopenmp','-lpcl_common','-lpcl_features','-lpcl_filters','-lpcl_io','-lpcl_kdtree','-lpcl_keypoints','-lpcl_octree','-lpcl_registration','-lpcl_sample_consensus ','-lpcl_search','-lpcl_segmentation','-lpcl_surface','-lboost_system'],include_dirs=['/usr/include/',
    	'/usr/include/pcl-1.7/','/usr/include/hdf5/serial/','/usr/include/eigen3/',
    	'/usr/include/vtk-6.2/','/usr/include/pcl-1.7/pcl/surface/'],extra_link_args=['-lgomp','/usr/local/lib/libpcl_common.so',
'/usr/local/lib/libpcl_features.so',
'/usr/local/lib/libpcl_filters.so',
'/usr/local/lib/libpcl_io.so',
'/usr/local/lib/libpcl_io_ply.so',
'/usr/local/lib/libpcl_kdtree.so',
'/usr/local/lib/libpcl_keypoints.so',
'/usr/local/lib/libpcl_octree.so',
'/usr/local/lib/libpcl_outofcore.so',
'/usr/local/lib/libpcl_people.so',
'/usr/local/lib/libpcl_recognition.so',
'/usr/local/lib/libpcl_registration.so',
'/usr/local/lib/libpcl_sample_consensus.so',
'/usr/local/lib/libpcl_search.so',
'/usr/local/lib/libpcl_segmentation.so',
'/usr/local/lib/libpcl_surface.so',
'/usr/local/lib/libpcl_tracking.so',
'/usr/local/lib/libpcl_visualization.so',
'/usr/lib/x86_64-linux-gnu/libboost_thread.so',
'/usr/lib/x86_64-linux-gnu/libpthread.so',
'/usr/lib/x86_64-linux-gnu/libboost_filesystem.so',
'/usr/lib/x86_64-linux-gnu/libboost_iostreams.so',
'/usr/lib/x86_64-linux-gnu/libboost_system.so',
'/usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5.so',
'/usr/lib/x86_64-linux-gnu/libvtkalglib-6.2.so',
'/usr/lib/x86_64-linux-gnu/libvtkDICOMParser-6.2.so',
'/usr/lib/x86_64-linux-gnu/libvtkexoIIc-6.2.so',
'/usr/lib/x86_64-linux-gnu/libvtkftgl-6.2.so',
'/usr/lib/x86_64-linux-gnu/libvtkmetaio-6.2.so',
'/usr/lib/x86_64-linux-gnu/libvtksys-6.2.so',
'/usr/lib/x86_64-linux-gnu/libvtkverdict-6.2.so',
'/usr/lib/x86_64-linux-gnu/libvtkCommonCore-6.2.so',
'/usr/lib/x86_64-linux-gnu/libvtkCommonMisc-6.2.so',
'/usr/lib/x86_64-linux-gnu/libvtkCommonColor-6.2.so',
'/usr/lib/x86_64-linux-gnu/libvtkCommonMath-6.2.so',
'/usr/local/lib/librestbed.a'])

setup(name="my_pcl", ext_modules = cythonize([ext]))




