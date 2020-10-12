#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
using namespace std;

#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/keypoints/harris_3d.h>
#include <pcl/search/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>

#include <boost/program_options.hpp>

int option = 0;

int parse_arguments(int argc, char *argv[]) {

	namespace po = boost::program_options;

	po::options_description desc("Allowed options");
	desc.add_options()
    ("help,h", "Please check comments in the main() function to see how to run the program..")
		("option,o", po::value<int> (&option), "1 - keypoint extraction, 2 - feature extraction 3 - correspondences");

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

  if(vm.count("help")) {
    cout << desc << endl;
    return -1;
  }
  if(!vm.count("option")) {
    cout << "Enter -o option.. Run ./test_features -h for details..\n";
    return -1;
  }
  return 0;
}

void insert_to_cmap(std::map<int, int>& cmap, int keyi, int mapi, vector<bool>& flag)
{
  pair< std::map<int,int>::iterator, bool > ret;
  
  ret = cmap.insert(make_pair(keyi, mapi));
  cmap.insert(make_pair(keyi, mapi));
  if(ret.second == false) {
    cout << "Key " << keyi << " already exists " << ret.first->first << " " << ret.first->second << " " << mapi << "..\n";
    flag[ret.first->second] = false;
  }
}
template<typename FeatureType>
void findCorrespondences (typename pcl::PointCloud<FeatureType>::Ptr source, 
                          typename pcl::PointCloud<FeatureType>::Ptr target, 
                          std::vector< pair<int, int> > & correspondences) {

  correspondences.reserve(source->size());

  pcl::KdTreeFLANN<FeatureType> descriptor_kdtree;
  descriptor_kdtree.setInputCloud (target);
  vector<bool> flags;
  flags.resize(source->points.size());
  for(int i=0; i<flags.size(); i++) {
    flags[i] = true;
  }

  const int k = 2;
  std::vector<int> k_indices (k);
  std::vector<float> k_squared_distances (k);
  std::map<int, int> cmap;
  for (size_t i = 0; i < source->size (); ++i) {
    descriptor_kdtree.nearestKSearch (*source, i, k, k_indices, k_squared_distances);
    int n1 = k_indices[0];
    int n2 = k_indices[1];

    //if( ( sqrt(k_squared_distances[0]) / sqrt(k_squared_distances[1]) ) > 0.4 ) {
    if( ( k_squared_distances[0] / k_squared_distances[1] ) > 0.4 ) {
      //correspondences.push_back(make_pair(i, k_indices[0]));
      insert_to_cmap(cmap, k_indices[0], i, flags);
    }
  }
  map<int, int>::iterator itr = cmap.begin();
  while(itr != cmap.end()) {
    if(flags[itr->second]) {
      cout << "Adding correspondence " << itr->second << " " << itr->first << endl;
      correspondences.push_back(make_pair(itr->second, itr->first));
    }
    itr++;
  }
  cout << "# of computed correspondences = " << cmap.size() << endl;
}
template<typename FeatureType>
void extractDescriptors (typename pcl::PointCloud<pcl::PointXYZ>::ConstPtr input, 
                         typename pcl::PointCloud<pcl::PointXYZI>::Ptr keypoints,
                         typename pcl::PointCloud<FeatureType>::Ptr features,
                         typename pcl::Feature<pcl::PointXYZ, FeatureType>::Ptr feature_extractor) {

  typename pcl::PointCloud<pcl::PointXYZ>::Ptr kpts(new pcl::PointCloud<pcl::PointXYZ>);
  kpts->points.resize(keypoints->points.size());
  
  pcl::copyPointCloud(*keypoints, *kpts);
          
  typename pcl::FeatureFromNormals<pcl::PointXYZ, pcl::Normal, FeatureType>::Ptr feature_from_normals = 
    boost::dynamic_pointer_cast<pcl::FeatureFromNormals<pcl::PointXYZ, pcl::Normal, FeatureType> > (feature_extractor);
  
  if (feature_from_normals) {
    cout << "Feature from normals\n";
    typename pcl::PointCloud<pcl::Normal>::Ptr normals (new  pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimation;
    normal_estimation.setSearchMethod (pcl::search::Search<pcl::PointXYZ>::Ptr (new pcl::search::KdTree<pcl::PointXYZ>));
    //normal_estimation.setRadiusSearch (5.0);
    normal_estimation.setRadiusSearch (0.01);
    normal_estimation.setInputCloud (input);
    normal_estimation.compute (*normals);
  
    feature_from_normals->setSearchSurface(input);
    feature_from_normals->setInputCloud(kpts);
    feature_from_normals->setInputNormals(normals);
    feature_from_normals->compute (*features);
  }
  else {
    cout << "Feature not from normals\n";
    feature_extractor->setSearchSurface(input);
    feature_extractor->setInputCloud(kpts);
    feature_extractor->compute (*features);
  }
}

void keyboard_callback (const pcl::visualization::KeyboardEvent& event, void* cookie) {

  pcl::visualization::PCLVisualizer *viewer = (pcl::visualization::PCLVisualizer*) cookie;
  if (event.keyUp()) {
    switch (event.getKeyCode()) {
      case 's':
      case 'S':
        static int index = 0;
        std::ostringstream ostr;
        ostr << "screen" << index++ << ".png";
        viewer->saveScreenshot(ostr.str());
        cout << "Saving screenshot " << ostr.str() << endl;
        break;
    }
  }
}
template <typename PointType, typename FeatureType>
void correspondences_test(char *argv[]) {

  typename pcl::PointCloud<pcl::PointXYZ>::Ptr source (new pcl::PointCloud<pcl::PointXYZ>());
  typename pcl::PointCloud<pcl::PointXYZ>::Ptr target (new pcl::PointCloud<pcl::PointXYZ>());

  typename pcl::PointCloud<PointType>::Ptr source_keypoints (new pcl::PointCloud<PointType>());
  typename pcl::PointCloud<PointType>::Ptr target_keypoints (new pcl::PointCloud<PointType>());

  typename pcl::PointCloud<FeatureType>::Ptr source_features (new pcl::PointCloud<FeatureType>); 
  typename pcl::PointCloud<FeatureType>::Ptr target_features (new pcl::PointCloud<FeatureType>);

  if(pcl::io::loadPCDFile (argv[1], *source) ) {
    cout << "Couldn't open file " << argv[1] << endl;
    return;
  }
  if(pcl::io::loadPCDFile (argv[2], *source_keypoints) ) {
    cout << "Couldn't open file " << argv[2] << endl;
    return;
  }
  if(pcl::io::loadPCDFile (argv[3], *source_features) ) {
    cout << "Couldn't open file " << argv[3] << endl;
    return;
  }
  if(pcl::io::loadPCDFile (argv[4], *target) ) {
    cout << "Couldn't open file " << argv[4] << endl;
    return;
  }
  if( pcl::io::loadPCDFile (argv[5], *target_keypoints) ) {
    cout << "Couldn't open file " << argv[5] << endl;
    return;
  }
  if(pcl::io::loadPCDFile (argv[6], *target_features) ) {
    cout << "Couldn't open file " << argv[6] << endl;
    return;
  }

  vector<pair< int, int> > correspondences;
  findCorrespondences<FeatureType> (source_features, target_features, correspondences);

  ofstream ofile("point-correspondences.txt");
  pcl::CorrespondencesPtr C ( new pcl::Correspondences);
  C->resize (correspondences.size());
  for (unsigned i = 0; i < correspondences.size(); i++) {
    (*C)[i].index_query = correspondences[i].first;
    (*C)[i].index_match = correspondences[i].second;
    ofile << correspondences[i].first << " " << correspondences[i].second << endl;

    //int c1 = correspondences[i].first;
    //int c2 = correspondences[i].second;

    //cout << " ( "  << source_keypoints->points[c1].x << " " << source_keypoints->points[c1].y 
    //      << source_keypoints->points[c1].z << " " << source_keypoints->points[c1].intensity << " ) <---> ";
    //cout << " ( " << target_keypoints->points[c2].x << " " << target_keypoints->points[c2].y 
    //      << target_keypoints->points[c2].z << " " << target_keypoints->points[c2].intensity << " ) " << endl;
  }
  ofile.close();

  //for(size_t i=0; i<correspondences.size(); i++) {
  //  cout << (*C)[i].index_query << " " << (*C)[i].index_match << endl;
  //}
  cout << "# of unique correspondences = " << correspondences.size() << endl;

  pcl::visualization::PCLVisualizer viewer;
  viewer.addPointCloud<pcl::PointXYZ>(source, "source");
  viewer.addPointCloud<pcl::PointXYZ>(target, "target");
  viewer.addCorrespondences<PointType>(source_keypoints, target_keypoints, *C, "correspondences");
  //viewer.addCorrespondences<pcl::PointXYZ>(source, target, *C, "correspondences");
  viewer.registerKeyboardCallback(keyboard_callback, &viewer);
  viewer.addCoordinateSystem (1.0);
  viewer.setBackgroundColor (0, 0, 0);
  viewer.initCameraParameters ();

	while (!viewer.wasStopped ()) {
		viewer.spinOnce (100);
	}
}

void harris_keypoints_test(char *argv[]) {

  pcl::PointCloud<pcl::PointXYZ>::Ptr source (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::io::loadPCDFile (argv[1], *source);
  pcl::PointCloud<pcl::PointXYZ>::Ptr target (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::io::loadPCDFile (argv[2], *target);

  pcl::HarrisKeypoint3D<pcl::PointXYZ,pcl::PointXYZI>* harris3D =
    new pcl::HarrisKeypoint3D<pcl::PointXYZ,pcl::PointXYZI> (pcl::HarrisKeypoint3D<pcl::PointXYZ, pcl::PointXYZI>::HARRIS);
  harris3D->setNonMaxSupression(true);

  harris3D->setRadius (0.01);
  //harris3D->setRadius (1.0);
  
  //harris3D->setThreshold(0.012);
  //harris3D->setThreshold(0.005);
  
  harris3D->setRadiusSearch (0.01);
  //harris3D->setRadiusSearch (1);

  //harris3D->setMethod(pcl::HarrisKeypoint3D<pcl::PointXYZ,pcl::PointXYZI>::TOMASI);
  harris3D->setMethod(pcl::HarrisKeypoint3D<pcl::PointXYZ,pcl::PointXYZI>::CURVATURE);

  boost::shared_ptr< pcl::Keypoint<pcl::PointXYZ, pcl::PointXYZI> > keypoint_detector;
  keypoint_detector.reset(harris3D);

  keypoint_detector->setInputCloud(source);
  pcl::PointCloud<pcl::PointXYZI>::Ptr source_keypoints (new pcl::PointCloud<pcl::PointXYZI> ());
  keypoint_detector->compute(*source_keypoints);
  cout << "Keypoints computed for the source.."<<source_keypoints->points.size()<<"\n";

  keypoint_detector->setInputCloud(target);
  pcl::PointCloud<pcl::PointXYZI>::Ptr target_keypoints (new pcl::PointCloud<pcl::PointXYZI> ());
  keypoint_detector->compute(*target_keypoints);
  cout << "Keypoints computed for the target.."<<target_keypoints->points.size()<<"\n";

  ostringstream ostr;
  ostr << "harris-keypoints-" << argv[1] ;
  cout << "Saving file " << ostr.str() << endl;
  pcl::io::savePCDFileASCII(ostr.str(), *source_keypoints);
  ostr.str("");
  ostr << "harris-keypoints-" << argv[2] ;
  cout << "Saving file " << ostr.str() << endl;
  pcl::io::savePCDFileASCII(ostr.str(), *target_keypoints);
}

void sift_keypoints_test(char *argv[]) {

  pcl::PointCloud<pcl::PointXYZ>::Ptr source (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::io::loadPCDFile (argv[1], *source);
  pcl::PointCloud<pcl::PointXYZ>::Ptr target (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::io::loadPCDFile (argv[2], *target);

  /*

  pcl::SIFTKeypoint<pcl::PointXYZ,pcl::PointXYZI>* sift = new pcl::SIFTKeypoint<pcl::PointXYZ,pcl::PointXYZI> ;
  
  sift->setScales(0.01, 3, 2);
  sift->setMinimumContrast(0.0);

  
  boost::shared_ptr< pcl::Keypoint<pcl::PointXYZ, pcl::PointXYZI> > keypoint_detector;
  keypoint_detector.reset(sift);

  keypoint_detector->setInputCloud(source);
  pcl::PointCloud<pcl::PointXYZI>::Ptr source_keypoints (new pcl::PointCloud<pcl::PointXYZI> ());
  keypoint_detector->compute(*source_keypoints);
  cout << "Keypoints computed for the source..\n";

  keypoint_detector->setInputCloud(target);
  pcl::PointCloud<pcl::PointXYZI>::Ptr target_keypoints (new pcl::PointCloud<pcl::PointXYZI> ());
  keypoint_detector->compute(*target_keypoints);
  cout << "Keypoints computed for the target..\n";

  pcl::io::savePCDFileASCII("sift_source_keypoints.pcd", *source_keypoints);
  pcl::io::savePCDFileASCII("sift_target_keypoints.pcd", *target_keypoints);
  */
}

template <typename DescriptorType>
void descriptors_test(char *argv[]) {

  pcl::PointCloud<pcl::PointXYZ>::Ptr source (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::io::loadPCDFile (argv[1], *source);

  pcl::PointCloud<pcl::PointXYZI>::Ptr source_keypoints (new pcl::PointCloud<pcl::PointXYZI> ());
  pcl::io::loadPCDFile (argv[2], *source_keypoints);

  typename pcl::Feature<pcl::PointXYZ, DescriptorType>::Ptr feature_extractor (new pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, DescriptorType>); 
  feature_extractor->setSearchMethod (pcl::search::Search<pcl::PointXYZ>::Ptr (new pcl::search::KdTree<pcl::PointXYZ>));
  feature_extractor->setRadiusSearch (0);
  feature_extractor->setKSearch (10);

  typename pcl::PointCloud<DescriptorType>::Ptr source_features (new pcl::PointCloud<DescriptorType>); 

  extractDescriptors<DescriptorType> (source, source_keypoints, source_features, feature_extractor);
  cout << "Descriptors computed for source..\n";
  cout << "# features = " << source_features->points.size() << endl;

  ostringstream ostr;
  ostr << "fpfh-" << argv[1];
  cout << "Saving file " << ostr.str() << endl;
  pcl::io::savePCDFileASCII(ostr.str(), *source_features);
  ostr.str("");

  pcl::PointCloud<pcl::PointXYZ>::Ptr target (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::io::loadPCDFile (argv[3], *target);

  pcl::PointCloud<pcl::PointXYZI>::Ptr target_keypoints (new pcl::PointCloud<pcl::PointXYZI> ());
  pcl::io::loadPCDFile (argv[4], *target_keypoints);

  typename pcl::PointCloud<DescriptorType>::Ptr target_features (new pcl::PointCloud<DescriptorType>); 

  extractDescriptors<DescriptorType> (target, target_keypoints, target_features, feature_extractor);
  cout << "Descriptors computed for target..\n";
  cout << "# features = " << target_features->points.size() << endl;
  ostr << "fpfh-" << argv[3];
  cout << "Saving file " << ostr.str() << endl;
  pcl::io::savePCDFileASCII(ostr.str(), *target_features);
}

int main(int argc, char *argv[]) {

  if(parse_arguments(argc, argv) == -1) {
    return 0;
  }

  if(option == 1) {
    // ./test_features source.pcd target.pcd
    harris_keypoints_test(argv);
  }

  if(option == 2) {
    // ./test_features source.pcd harris_source_keypoints.pcd target.pcd harris_target_keypoints.pcd
    descriptors_test<pcl::FPFHSignature33>(argv);
  }

  if(option == 3) {
    // ./test_features source.pcd harris_source_keypoints.pcd source_fpfh.pcd target.pcd harris_target_keypoints.pcd target_fpfh.pcd
    correspondences_test<pcl::PointXYZI, pcl::FPFHSignature33> (argv);
  }

  return 0;
}	
