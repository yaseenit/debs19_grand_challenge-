#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/vfh.h>
#include <pcl/common/geometry.h>
#include <pcl/features/esf.h>
#include <dirent.h>
#include <memory>
#include <cstdlib>
#include <restbed>
#include <stdio.h>

using namespace std;
using namespace restbed;



void computeVFH(std::string pcdFilePath, float radius);
void computeESF(std::string pcdFilePath);
pcl::PointCloud<pcl::PointXYZ>::Ptr   build_pointcloud(string pointCloudAsText);


bool replace(std::string& str, const std::string& from, const std::string& to) {
    size_t start_pos = str.find(from);
    if(start_pos == std::string::npos)
        return false;
    str.replace(start_pos, from.length(), to);
    return true;
}



int main (int argc, char **argv)
{

    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir (argv[1])) != NULL)
    {
        /* list all the files and directories within directory */
        while ((ent = readdir (dir)) != NULL)
        {

            char *isPcdFile = NULL;
            isPcdFile = strstr (ent->d_name, ".pcd");// filter out not pcd files
            if(isPcdFile)
            {
               std::string inputFilepath(argv[1]);
               inputFilepath.append(ent->d_name);
               //computeVFH(inputFilepath, atof(argv[2]));
               computeESF(inputFilepath);
            }
        }//end of while
        closedir (dir);
    }
    else
    {
        /* could not open directory */
        perror ("could not open directory");
        return EXIT_FAILURE;
    }
    return 0;
}

void computeVFH(std::string pcdFilePath, float radius)
{

    // Read in the cloud data
    pcl::PCDReader reader;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>), cloud_f (new pcl::PointCloud<pcl::PointXYZ>);
    reader.read (pcdFilePath, *cloud);
    //std::cout << "PointCloud has: " << cloud->points.size () << " data points." << std::endl;


    // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);

    // Create the normal estimation class, and pass the input dataset to it
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud (cloud);

    // Create an empty kdtree representation, and pass it to the normal estimation object.
    // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
    ne.setSearchMethod (tree);

    // Output datasets
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);

    // Use all neighbors in a sphere of radius 3cm
    ne.setRadiusSearch (radius); //0.03 is 3 cm atof(argv[1])

    // Compute the features
    ne.compute (*cloud_normals);


    // Create the VFH estimation class, and pass the input dataset+normals to it
    pcl::VFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::VFHSignature308> vfh;
    vfh.setInputCloud (cloud);
    vfh.setInputNormals (cloud_normals);
    vfh.setSearchMethod (tree);

    // Output datasets
    pcl::PointCloud<pcl::VFHSignature308>::Ptr vfhs (new pcl::PointCloud<pcl::VFHSignature308> ());

    // Compute the features
    vfh.compute (*vfhs);

    replace(pcdFilePath, ".pcd", "");

    std::string outputFilePath = std::string(pcdFilePath) + ".vfh";
    pcl::io::savePCDFileASCII (outputFilePath, *vfhs);

}


void computeESF(std::string pcdFilePath){


    //std::cout << pcdFilePath << std::endl;



	// Cloud for storing the object.
	pcl::PointCloud<pcl::PointXYZ>::Ptr object(new pcl::PointCloud<pcl::PointXYZ>);
	// Object for storing the ESF descriptor.
	pcl::PointCloud<pcl::ESFSignature640>::Ptr descriptor(new pcl::PointCloud<pcl::ESFSignature640>);

	// Note: you should have performed preprocessing to cluster out the object
	// from the cloud, and save it to this individual file.

	// Read a PCD file from disk.
	if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcdFilePath, *object) != 0)
	{
        perror ("could not open pcd file!");
	}

	// ESF estimation object.
	pcl::ESFEstimation<pcl::PointXYZ, pcl::ESFSignature640> esf;
	esf.setInputCloud(object);

	esf.compute(*descriptor);

    replace(pcdFilePath, ".pcd", "");

    std::string outputFilePath = std::string(pcdFilePath) + ".esf";
    pcl::io::savePCDFileASCII (outputFilePath, *descriptor);
}




