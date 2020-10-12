#include <iostream>
#include <pcl/filters/voxel_grid.h>
#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>

#include <pcl/features/fpfh.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/normal_3d.h>
#include<pcl/visualization/pcl_plotter.h>
#include <pcl/point_types.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection.h>

#include <vector>
#include <string>
#include <sstream>
#include <fstream>
using namespace std;

#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/search/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <boost/program_options.hpp>


#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>


#include <pcl/segmentation/region_growing.h>
#include <pcl/visualization/cloud_viewer.h>

#define LEAFSIZE   2.0f,1.8f,0.9f,0.0f 







boost::shared_ptr<pcl::visualization::PCLVisualizer> simpleVis (pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addPointCloud<pcl::PointXYZ> (cloud, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
  viewer->addCoordinateSystem (0.2);
  viewer->initCameraParameters ();
  return (viewer);
}

boost::shared_ptr<pcl::visualization::PCLVisualizer> normalsVis (
    pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud, pcl::PointCloud<pcl::Normal>::ConstPtr normals)
{
  // --------------------------------------------------------
  // -----Open 3D viewer and add point cloud and normals-----
  // --------------------------------------------------------
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  //pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
  viewer->addPointCloud<pcl::PointXYZ> (cloud, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
  viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal> (cloud, normals, 10, 9.5, "normals");
  //viewer->addCoordinateSystem (1.0);
  //viewer->initCameraParameters ();
  return (viewer);
}




unsigned int text_id = 0;
void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event,
                            void* viewer_void)
{
  pcl::visualization::PCLVisualizer *viewer = static_cast<pcl::visualization::PCLVisualizer *> (viewer_void);
  if (event.getKeySym () == "r" && event.keyDown ())
  {
    std::cout << "r was pressed => removing all text" << std::endl;

    char str[512];
    for (unsigned int i = 0; i < text_id; ++i)
    {
      sprintf (str, "text#%03d", i);
      viewer->removeShape (str);
    }
    text_id = 0;
  }
}

void mouseEventOccurred (const pcl::visualization::MouseEvent &event,
                         void* viewer_void)
{
  pcl::visualization::PCLVisualizer *viewer = static_cast<pcl::visualization::PCLVisualizer *> (viewer_void);
  if (event.getButton () == pcl::visualization::MouseEvent::LeftButton &&
      event.getType () == pcl::visualization::MouseEvent::MouseButtonRelease)
  {
    std::cout << "Left mouse button released at position (" << event.getX () << ", " << event.getY () << ")" << std::endl;

    char str[512];
    sprintf (str, "text#%03d", text_id ++);
    viewer->addText ("clicked here", event.getX (), event.getY (), str);
  }
}

boost::shared_ptr<pcl::visualization::PCLVisualizer> interactionCustomizationVis ()
{
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addCoordinateSystem (1.0);

  viewer->registerKeyboardCallback (keyboardEventOccurred, (void*)viewer.get ());
  viewer->registerMouseCallback (mouseEventOccurred, (void*)viewer.get ());

  return (viewer);
}


// --------------
// -----Main-----
// --------------
int
main (int argc, char** argv)
{
  // --------------------------------------
  // -----Parse Command Line Arguments-----
  // --------------------------------------
  if (pcl::console::find_argument (argc, argv, "-h") >= 0)
  {
    
  }
  bool simple(false), nor(false), cor(false);
  if (pcl::console::find_argument (argc, argv, "-s") >= 0)
  {
    simple = true;
    std::cout << "Simple visualisation example\n";
  }

else if (pcl::console::find_argument (argc, argv, "-n") >= 0)
  {
    nor = true;
    std::cout << "Normals visualisation example\n";
  }
else if (pcl::console::find_argument (argc, argv, "-c") >= 0)
  {
    cor = true;
    std::cout << "Correspondence visualisation example\n";
  }
  else
  {
    //printUsage (argv[0]);
    return 0;
  }

  // ------------------------------------
  // -----Create example point cloud-----
  // ------------------------------------
 

 pcl::PointCloud<pcl::PointXYZ>::Ptr basic_cloud_ptr (new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr basic_cloud_ptr1 (new pcl::PointCloud<pcl::PointXYZ>); 
 // Object for storing the downsampled data.
 pcl::PointCloud<pcl::PointXYZ>::Ptr filteredCloud(new pcl::PointCloud<pcl::PointXYZ>);
 pcl::PointCloud<pcl::PointXYZ>::Ptr filteredCloud1(new pcl::PointCloud<pcl::PointXYZ>);
 // Object for storing the normals.
 pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);

//reading pcd 
	pcl::PCDReader reader;
   	reader.read<pcl::PointXYZ> ("michael0.pcd", *basic_cloud_ptr);
   	reader.read<pcl::PointXYZ> ("michael8.pcd", *basic_cloud_ptr1);


	//DOWNSAMPLE
	// Filter object.
	pcl::VoxelGrid<pcl::PointXYZ> filter;
	filter.setInputCloud(basic_cloud_ptr);
	// We set the size of every voxel to be 1x1x1cm
	// (only one point per every cubic centimeter will survive).
	filter.setLeafSize(1.56f, 1.56f, 1.56f);
	filter.filter(*filteredCloud);

	//DOWNSAMPLE
	// Filter object.
	pcl::VoxelGrid<pcl::PointXYZ> filter1;
	filter1.setInputCloud(basic_cloud_ptr1);
	// We set the size of every voxel to be 1x1x1cm
	// (only one point per every cubic centimeter will survive).
	filter1.setLeafSize(1.56f, 1.56f, 1.56f);
	filter1.filter(*filteredCloud1);

pcl::PCDWriter writer;
	writer.write<pcl::PointXYZ> ("downsample.pcd", *filteredCloud, false);
	writer.write<pcl::PointXYZ> ("downsample1.pcd", *filteredCloud1, false);

///////////////////////////////////////////////////////////////
//NORMAL ESTIMATION
/*		
	// Object for normal estimation.
	pcl::NormalEstimation<pcl::PointXYZ, pcl::PointXYZ> normalEstimation;
	normalEstimation.setInputCloud(filteredCloud);
	// For every point, use all neighbors in a radius of 3cm.
	normalEstimation.setRadiusSearch(0.03);
	// A kd-tree is a data structure that makes searches efficient. More about it later.
	// The normal estimation object will use it to find nearest neighbors.
	pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
	normalEstimation.setSearchMethod(kdtree);

	// Calculate the normals.
	normalEstimation.compute(*normals);

	// Visualize them.


	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Normals"));
	viewer->addPointCloud<pcl::PointXYZ>(basic_cloud_ptr, "cloud");
	// Display one normal out of 20, as a line of length 3cm.
	viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(basic_cloud_ptr, normals, 20, 0.03, "normals");
	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}


*/





// ----------------------------------------------------------------
  // -----Calculate surface normals with a search radius of 0.05 for 1st pcd-----
  // ----------------------------------------------------------------
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
  ne.setInputCloud (filteredCloud);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
  ne.setSearchMethod (tree);
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals1 (new pcl::PointCloud<pcl::Normal>);
  ne.setRadiusSearch (2.0);
  //ne.setKSearch(4); 
  ne.compute (*cloud_normals1);

  // ---------------------------------------------------------------
  // -----Calculate surface normals with a search radius of 0.05 for 2nd pcd-----
  // ---------------------------------------------------------------
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne1;
  ne1.setInputCloud (filteredCloud1);
  ne1.setSearchMethod (tree);
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals2 (new pcl::PointCloud<pcl::Normal>);
  ne1.setRadiusSearch (2.0);
  //ne1.setKSearch(4); 
  ne1.compute (*cloud_normals2);

writer.write<pcl::Normal> ("normal.pcd", *cloud_normals1, false);
writer.write<pcl::Normal> ("normal1.pcd", *cloud_normals2, false);


///////////////////////////////////////////////////////
//FPFH

//FPFH for 1st pcd	
// Create the FPFH estimation class, and pass the input dataset+normals to it
	pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
	fpfh.setInputCloud (filteredCloud);
	fpfh.setInputNormals (cloud_normals1);
	// Create an empty kdtree representation, and pass it to the FPFH estimation object.
	// Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree1 (new pcl::search::KdTree<pcl::PointXYZ> ());
	fpfh.setSearchMethod (tree1);
 // Output datasets
  pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs (new pcl::PointCloud<pcl::FPFHSignature33> ());

  // Use all neighbors in a sphere of radius 5cm
  // IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!
  fpfh.setRadiusSearch (0);
  fpfh.setKSearch(100);

  // Compute the features
  fpfh.compute (*fpfhs);
  pcl::visualization::PCLPlotter plotter;
  plotter.addFeatureHistogram(*fpfhs,33);
  plotter.plot(); 
   // Index idx = x for displaying the histogram corresponding to point P_x with index x from the original point cloud
        int idx = 1000;
// display values of FPFH for Point P_1000
        for (int j = 0; j < fpfhs->points[idx].descriptorSize(); j++ )
                cout << endl << fpfhs->points[idx].histogram[j] << endl;
writer.write<pcl::FPFHSignature33> ("fpfh.pcd", *fpfhs, false);

//FPFH for 2nd pcd
// Create the FPFH estimation class, and pass the input dataset+normals to it
	pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh1;
	fpfh1.setInputCloud (filteredCloud1);
	fpfh1.setInputNormals (cloud_normals2);
	// Create an empty kdtree representation, and pass it to the FPFH estimation object.
	// Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree2 (new pcl::search::KdTree<pcl::PointXYZ> ());
	fpfh1.setSearchMethod (tree2);
 // Output datasets
  pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs1 (new pcl::PointCloud<pcl::FPFHSignature33> ());

  // Use all neighbors in a sphere of radius 5cm
  // IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!
  fpfh1.setRadiusSearch (0);
  fpfh1.setKSearch(100);

  // Compute the features
  fpfh1.compute (*fpfhs1);
  pcl::visualization::PCLPlotter plotter1;
  plotter1.addFeatureHistogram(*fpfhs1,33);
plotter1.addFeatureHistogram(*fpfhs,33);
  plotter1.plot(); 
   // Index idx = x for displaying the histogram corresponding to point P_x with index x from the original point cloud
        idx = 1000;
// display values of FPFH for Point P_1000
        for (int j = 0; j < fpfhs1->points[idx].descriptorSize(); j++ )
                cout << endl << fpfhs1->points[idx].histogram[j] << endl;
writer.write<pcl::FPFHSignature33> ("fpfh1.pcd", *fpfhs1, false);



  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
  if (simple)
  {
    viewer = simpleVis(basic_cloud_ptr);
  }
  else if (nor)
  {
    viewer = normalsVis(filteredCloud, cloud_normals1);
  }

else if (cor)
	{

 		PCL_INFO ("Correspondence Estimation\n"); 
                pcl::registration::CorrespondenceEstimation<pcl::FPFHSignature33, pcl::FPFHSignature33> corEst; 
                corEst.setInputSource(fpfhs1); 
                corEst.setInputTarget (fpfhs); 
                PCL_INFO (" Correspondence Estimation - Estimate C.\n"); 
                //pcl::Correspondences cor_all; 
                //Pointer erzeugen 
                boost::shared_ptr<pcl::Correspondences> cor_all_ptr (new pcl::Correspondences); 
		boost::shared_ptr<pcl::Correspondences> cor_all_ptr_reciprocal (new pcl::Correspondences);
                corEst.determineCorrespondences (*cor_all_ptr);	
		corEst.determineReciprocalCorrespondences(*cor_all_ptr_reciprocal);
                PCL_INFO (" Correspondence Estimation - Found %d Correspondences\n", cor_all_ptr->size());
		PCL_INFO (" Reciprocal Correspondence Estimation - Found %d Correspondences\n", cor_all_ptr_reciprocal->size());
// Visualizer routine 
boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer
(new pcl::visualization::PCLVisualizer ("Sparse reciprocal correspondence"));
viewer->addPointCloud (basic_cloud_ptr,  "moved_src", 0); 
viewer->addPointCloud (basic_cloud_ptr1, "moved_tgt", 0); 
//viewer->addPointCloud (cloud_normals1,  "key_src", 0); 
//viewer->addPointCloud (cloud_normals2,"key_tgt", 0); 
                        
viewer->addCorrespondences<pcl::PointXYZ> (basic_cloud_ptr, basic_cloud_ptr1, *cor_all_ptr_reciprocal, "correspondences", 0); 
while (!viewer->wasStopped ())
  {
    viewer->spinOnce (100);
    boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  }

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//RANSAC
    pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZ> corrsRejectorSAC; 
    corrsRejectorSAC.setInputSource(filteredCloud1); 
    corrsRejectorSAC.setInputTarget(filteredCloud); 
    corrsRejectorSAC.setInlierThreshold(3.0); 
	//double x=corrsRejectorSAC.getInlierThreshold();	
 	//std::cout <<"threshhold:"<< x << std::endl;
    corrsRejectorSAC.setMaximumIterations(10000); 
    corrsRejectorSAC.setInputCorrespondences(cor_all_ptr_reciprocal); 
    boost::shared_ptr<pcl::Correspondences> correspondencesRejSAC (new pcl::Correspondences); 
    corrsRejectorSAC.getCorrespondences(*correspondencesRejSAC); 
    Eigen::Matrix4f transformation=corrsRejectorSAC.getBestTransformation(); 
 std::cout << transformation << std::endl;

pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud (new pcl::PointCloud<pcl::PointXYZ> ());
  // You can either apply transform_1 or transform_2; they are the same
  pcl::transformPointCloud (*filteredCloud1, *transformed_cloud, transformation);

// Visualization of RANSAC
  printf(  "\nPoint cloud colors :  white  = original point cloud\n"
      "                        red  = transformed point cloud\n");
  pcl::visualization::PCLVisualizer viewer4 ("Initial Allignment");

   // Define R,G,B colors for the point cloud
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_cloud_color_handler (basic_cloud_ptr, 255, 255, 255);
  // We add the point cloud to the viewer and pass the color handler
  viewer4.addPointCloud<pcl::PointXYZ>(filteredCloud, source_cloud_color_handler, "original_cloud");

  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> transformed_cloud_color_handler (transformed_cloud, 230, 20, 20); // Red
  viewer4.addPointCloud<pcl::PointXYZ>(transformed_cloud, transformed_cloud_color_handler, "transformed_cloud");

  viewer4.addCoordinateSystem (1.0, "cloud", 0);
  viewer4.setBackgroundColor(0.05, 0.05, 0.05, 0); // Setting background to a dark grey
  viewer4.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "original_cloud");
  viewer4.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "transformed_cloud");
  //viewer.setPosition(800, 400); // Setting visualiser window position

  while (!viewer4.wasStopped ()) { // Display the visualiser until 'q' key is pressed
    viewer4.spinOnce ();
  }

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// concatenating two point clouds
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_c(new pcl::PointCloud<pcl::PointXYZ>()) ;
*cloud_c  = *filteredCloud;
    *cloud_c += *transformed_cloud;
    std::cerr << "Cloud C " << std::endl;
writer.write<pcl::PointXYZ> ("transformed_cloud.pcd", *transformed_cloud, false);
writer.write<pcl::PointXYZ> ("align.pcd", *cloud_c, false);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Correspondence aftere initial allignment



//FPFH for transformed pcd

//normals for transfomred cloud
 pcl::search::Search<pcl::PointXYZ>::Ptr tree = boost::shared_ptr<pcl::search::Search<pcl::PointXYZ> > (new pcl::search::KdTree<pcl::PointXYZ>);
  pcl::PointCloud <pcl::Normal>::Ptr normalsa (new pcl::PointCloud <pcl::Normal>);
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
  normal_estimator.setSearchMethod (tree);
  normal_estimator.setInputCloud (transformed_cloud);
  normal_estimator.setKSearch (50);
  normal_estimator.compute (*normalsa);

// Create the FPFH estimation class, and pass the input dataset+normals to it
	pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh_Transformed;
	fpfh_Transformed.setInputCloud (transformed_cloud);
	fpfh_Transformed.setInputNormals (normalsa);
	// Create an empty kdtree representation, and pass it to the FPFH estimation object.
	// Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
	//pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (ne pcl::search::KdTree<pcl::PointXYZ> ());
	fpfh_Transformed.setSearchMethod (tree);
 // Output datasets
  pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs1_transformed (new pcl::PointCloud<pcl::FPFHSignature33> ());

  // Use all neighbors in a sphere of radius 5cm
  // IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!
  fpfh_Transformed.setRadiusSearch (0);
  fpfh_Transformed.setKSearch(100);

  // Compute the features
  fpfh_Transformed.compute (*fpfhs1_transformed);
  pcl::visualization::PCLPlotter plotter1;
  plotter1.addFeatureHistogram(*fpfhs,33);
  plotter1.addFeatureHistogram(*fpfhs1,33);
  plotter1.addFeatureHistogram(*fpfhs1_transformed,33);
  plotter1.plot(); 
   // Index idx = x for displaying the histogram corresponding to point P_x with index x from the original point cloud
        idx = 1000;
// display values of FPFH for Point P_1000
        for (int j = 0; j < fpfhs1_transformed->points[idx].descriptorSize(); j++ )
                cout << endl << fpfhs1_transformed->points[idx].histogram[j] << endl;
writer.write<pcl::FPFHSignature33> ("fpfh_Transformed.pcd", *fpfhs1_transformed, false);

//Correspondence b/w source and transformed/alligned cloud
		PCL_INFO ("Correspondence Estimation\n"); 
                pcl::registration::CorrespondenceEstimation<pcl::FPFHSignature33, pcl::FPFHSignature33> corEst1; 
                corEst1.setInputSource(fpfhs); 
                corEst1.setInputTarget (fpfhs1_transformed); 
                PCL_INFO (" Correspondence Estimation - Estimate C.\n"); 
                //pcl::Correspondences cor_all; 
                //Pointer erzeugen 
                boost::shared_ptr<pcl::Correspondences> cor_all_ptr1 (new pcl::Correspondences); 
		boost::shared_ptr<pcl::Correspondences> cor_all_ptr_reciprocal1 (new pcl::Correspondences);
                corEst1.determineCorrespondences (*cor_all_ptr1);	
		corEst1.determineReciprocalCorrespondences(*cor_all_ptr_reciprocal1);
                PCL_INFO (" Correspondence Estimation - Found %d Correspondences\n", cor_all_ptr1->size());
		PCL_INFO (" Reciprocal Correspondence Estimation - Found %d Correspondences\n", cor_all_ptr_reciprocal1->size());
// Visualizer routine 
boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer8
(new pcl::visualization::PCLVisualizer ("correspondences_after_initial_allignment"));
viewer8->addPointCloud (basic_cloud_ptr,  "moved_s", 0); 
viewer8->addPointCloud (transformed_cloud, "moved_t", 0); 
//viewer->addPointCloud (cloud_normals1,  "key_src", 0); 
//viewer->addPointCloud (cloud_normals2,"key_tgt", 0); 
                        
viewer8->addCorrespondences<pcl::PointXYZ> (basic_cloud_ptr, transformed_cloud, *cor_all_ptr_reciprocal1, "correspondences_after_initial_allignment", 0); 
while (!viewer8->wasStopped ())
  {
    viewer8->spinOnce (100);
    boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  }




//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//SEGMENTATION

viewer = normalsVis(transformed_cloud, normalsa);
while (!viewer->wasStopped ())
  {
    viewer->spinOnce (100);
    boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  }

pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> reg;
  reg.setMinClusterSize (100);
  reg.setMaxClusterSize (10000);
  reg.setSearchMethod (tree);
  reg.setNumberOfNeighbours (30);
  reg.setInputCloud (transformed_cloud);
  //reg.setIndices (indices);
  reg.setInputNormals (normalsa);
  reg.setSmoothnessThreshold (7.0 / 180.0 * M_PI);
  reg.setCurvatureThreshold (0.05);

  std::vector <pcl::PointIndices> clusters;
  reg.extract (clusters);

std::cout << "Number of clusters is equal to " << clusters.size () << std::endl;
  std::cout << "First cluster has " << clusters[0].indices.size () << " points." << endl;
  std::cout << "These are the indices of the points of the initial" <<
    std::endl << "cloud that belong to the first cluster:" << std::endl;
  int counter = 0;
  while (counter < clusters[0].indices.size ())
  {
    std::cout << clusters[0].indices[counter] << ", ";
    counter++;
    if (counter % 10 == 0)
      std::cout << std::endl;
  }
  std::cout << std::endl;

  pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud ();
  pcl::visualization::CloudViewer viewer6 ("Cluster viewer");
  viewer6.showCloud(colored_cloud);
  while (!viewer6.wasStopped ())
  {
  }


///////////////////////////////
}
  //--------------------
  // -----Main loop-----
  //--------------------
  while (!viewer->wasStopped ())
  {
    viewer->spinOnce (100);
    boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  }
}
