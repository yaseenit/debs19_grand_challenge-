/*
Passing variables / arrays between cython and cpp
Example from 
http://docs.cython.org/src/userguide/wrapping_CPlusPlus.html

Adapted to include passing of multidimensional arrays

*/

#include "Wrapper.h"
#include <pcl/features/esf.h>
#include <pcl/io/pcd_io.h>

using namespace pclwrapper;

Wrapper::Wrapper()
{

}

Wrapper::~Wrapper()
{
}



/*
Inputting a 2D vector, performing a simple operation and returning a new 2D vector
*/
std::vector<long double>  Wrapper::compute(std::vector< std::vector<double> > sv)
{

 int svrows = sv.size();
 int svcols = sv[0].size();

 std::vector< std::vector<double> > tot;
 tot.resize(svrows, std::vector<double> (svcols, -1));


// std::cout << "vector length " << svrows << " , " << svcols << std::endl;


 pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);



 // Fill in the cloud data
 cloud->width  = svrows;
 cloud->height = 1;
 cloud->points.resize (cloud->width * cloud->height);

 // Note: you should have performed preprocessing to cluster out the object
 // from the cloud, and save it to this individual file.
 for (int ii=0; ii<svrows; ii++)
 {

     cloud->points[ii].x = sv.at(ii).at(0);
    cloud->points[ii].y = sv.at(ii).at(1);
     cloud->points[ii].z = sv.at(ii).at(2);



 }        




// Object for storing the ESF descriptor.
 pcl::PointCloud<pcl::ESFSignature640>::Ptr descriptor(new pcl::PointCloud<pcl::ESFSignature640>);




 // ESF estimation object.
 pcl::ESFEstimation<pcl::PointXYZ, pcl::ESFSignature640> esf;
 esf.setInputCloud(cloud);

esf.compute(*descriptor);


 //std::string outputFilePath = "dummy.esf";
 //pcl::io::savePCDFileASCII (outputFilePath, *descriptor);


 std::vector<long double> histogram;
histogram.resize(pcl::ESFSignature640::descriptorSize ());
   for (int i = 0; i < pcl::ESFSignature640::descriptorSize (); i++) 
     { 
        histogram[i] = descriptor->points[0].histogram[i]; 
     } 


return histogram;


}