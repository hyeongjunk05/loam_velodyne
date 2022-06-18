// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// This is an implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

#include <cmath>

#include <loam_velodyne/common.h>
#include <nav_msgs/Odometry.h>
#include <opencv/cv.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h> // kd-tree는 library로 하네
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

// #include <vector> // scanRegistration에는 이것만 더 있고, laserOdometry.cpp 랑 include들은 같음.

// voide 짧은 거로 10개 있고, main 600줄 짜리 존재. 

//One point cloud cycle
const float scanPeriod = 0.1;

//The number of frames skipped, controlling the frequency sent to laserMapping
const int skipFrameNum = 1;
bool systemInited = false;

//Timestamp information
double timeCornerPointsSharp = 0;
double timeCornerPointsLessSharp = 0;
double timeSurfPointsFlat = 0;
double timeSurfPointsLessFlat = 0;
double timeLaserCloudFullRes = 0;
double timeImuTrans = 0;

//Message receiving flag
bool newCornerPointsSharp = false;
bool newCornerPointsLessSharp = false;
bool newSurfPointsFlat = false;
bool newSurfPointsLessFlat = false;
bool newLaserCloudFullRes = false;
bool newImuTrans = false;

//receive sharp points
pcl::PointCloud<PointType>::Ptr cornerPointsSharp(new pcl::PointCloud<PointType>()); //Ptr의미 머임?
//receive less sharp points
pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp(new pcl::PointCloud<PointType>());
//receive flat points
pcl::PointCloud<PointType>::Ptr surfPointsFlat(new pcl::PointCloud<PointType>());
//receive less flat points
pcl::PointCloud<PointType>::Ptr surfPointsLessFlat(new pcl::PointCloud<PointType>());

//less sharp points of last frame
pcl::PointCloud<PointType>::Ptr laserCloudCornerLast(new pcl::PointCloud<PointType>());
//less flat points of last frame
pcl::PointCloud<PointType>::Ptr laserCloudSurfLast(new pcl::PointCloud<PointType>());

//Save the unprocessed feature points sent by the previous node
pcl::PointCloud<PointType>::Ptr laserCloudOri(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointType>::Ptr coeffSel(new pcl::PointCloud<PointType>());

//receive all points
pcl::PointCloud<PointType>::Ptr laserCloudFullRes(new pcl::PointCloud<PointType>()); // Res 는 무슨 의미?
//receive imu info
pcl::PointCloud<pcl::PointXYZ>::Ptr imuTrans(new pcl::PointCloud<pcl::PointXYZ>());

//kd-tree built by less sharp points of last frame
pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerLast(new pcl::KdTreeFLANN<PointType>()); // FLANN의 의미?
//kd-tree built by less flat points of last frame
pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfLast(new pcl::KdTreeFLANN<PointType>());

int laserCloudCornerLastNum;
int laserCloudSurfLastNum;

//unused
int pointSelCornerInd[40000];
//save 2 corner points index searched
float pointSearchCornerInd1[40000];
float pointSearchCornerInd2[40000];

//unused
int pointSelSurfInd[40000];
//save 3 surf points index searched
float pointSearchSurfInd1[40000];
float pointSearchSurfInd2[40000];
float pointSearchSurfInd3[40000];

//The amount of state transition between the current frame and the previous frame，in the local frame // state transition!
float transform[6] = {0}; // 배열 선언!
//The amount of state transition of the current frame relative to the first frame，in the global frame // state transition! relative to the first frame! in the global frame!
float transformSum[6] = {0};

//RPY of the first point of the point cloud
float imuRollStart = 0, imuPitchStart = 0, imuYawStart = 0;
//RPY of the last point of the point cloud
float imuRollLast = 0, imuPitchLast = 0, imuYawLast = 0;
//Distortion displacement of the last point of the point cloud relative to the first point due to acceleration and deceleration
float imuShiftFromStartX = 0, imuShiftFromStartY = 0, imuShiftFromStartZ = 0;
//The distortion speed of the last point of the point cloud relative to the first point due to acceleration and deceleration
float imuVeloFromStartX = 0, imuVeloFromStartY = 0, imuVeloFromStartZ = 0;

/*****************************************************************************
    The role of the current frame point cloud TransformToStart and the previous frame point cloud TransformToEnd：
         Remove the distortion, and unify the two frames of point cloud data into the same coordinate system for calculation

         여기부터 ~ line 247까지
*****************************************************************************/

//The point in the current point cloud is relative to the first point to remove the distortion caused by the uniform motion, the effect is equivalent to the point cloud obtained by the static scan at the start position of the point cloud scan
void TransformToStart(PointType const * const pi, PointType * const po) // main f line 551, 691 쯤에서 쓰임. pi와 po는 여기서 정의됨
{
  //Interpolation coefficient calculation, the relative time of each point in the cloud/point cloud period 10
  float s = 10 * (pi->intensity - int(pi->intensity)); // pi 뭐여? 위에서 pi가 정의되지는 않았네. 라이브러리에 있는거도 아님

  //Linear interpolation: According to the relative position of each point in the point cloud, multiply it by the corresponding rotation and translation coefficient
  float rx = s * transform[0]; // 여기가 rotation & translation에 대해서 linear interpolation 하는 코드군!
  float ry = s * transform[1]; // 이게 0부터 5까지 rotation들과 translation 들에 대해서 보간법하는 거임
  float rz = s * transform[2];
  float tx = s * transform[3];
  float ty = s * transform[4];
  float tz = s * transform[5];

  //Rotate around the z axis after translation (-rz)
  float x1 = cos(rz) * (pi->x - tx) + sin(rz) * (pi->y - ty);
  float y1 = -sin(rz) * (pi->x - tx) + cos(rz) * (pi->y - ty);
  float z1 = (pi->z - tz);

  //Rotate around the x axis（-rx）
  float x2 = x1;
  float y2 = cos(rx) * y1 + sin(rx) * z1;
  float z2 = -sin(rx) * y1 + cos(rx) * z1;

  //Rotate around the y axis（-ry）
  po->x = cos(ry) * x2 - sin(ry) * z2;
  po->y = y2;
  po->z = sin(ry) * x2 + cos(ry) * z2;
  po->intensity = pi->intensity;
}

//Remove the distortion caused by the uniform motion of the point in the last frame of the point cloud relative to the end position, the effect is equivalent to the point cloud obtained by the static scan at the end position of the point cloud scan
void TransformToEnd(PointType const * const pi, PointType * const po) // line 1017 부근 이후 에서 쓰임
{
  //Interpolation coefficient calculation
  float s = 10 * (pi->intensity - int(pi->intensity));

  float rx = s * transform[0]; // 맨 처음에는 초기값이 0이므로 모두 0
  float ry = s * transform[1];
  float rz = s * transform[2];
  float tx = s * transform[3];
  float ty = s * transform[4];
  float tz = s * transform[5];

  //Rotate around the z axis after translation（-rz）
  float x1 = cos(rz) * (pi->x - tx) + sin(rz) * (pi->y - ty);
  float y1 = -sin(rz) * (pi->x - tx) + cos(rz) * (pi->y - ty);
  float z1 = (pi->z - tz);

  //Rotate around the x axis（-rx)
  float x2 = x1;
  float y2 = cos(rx) * y1 + sin(rx) * z1;
  float z2 = -sin(rx) * y1 + cos(rx) * z1;

  //Rotate around the y axis（-ry）
  float x3 = cos(ry) * x2 - sin(ry) * z2;
  float y3 = y2;
  float z3 = sin(ry) * x2 + cos(ry) * z2;//Calculate the coordinate corrected relative to the starting point

  rx = transform[0]; // 위에 178 line에서 선언해서 필요 없음 얘네들.
  ry = transform[1];
  rz = transform[2];
  tx = transform[3];
  ty = transform[4];
  tz = transform[5];

  //Rotate around the y axis（ry）
  float x4 = cos(ry) * x3 + sin(ry) * z3;
  float y4 = y3;
  float z4 = -sin(ry) * x3 + cos(ry) * z3;

  //Rotate around the x axis（rx）
  float x5 = x4;
  float y5 = cos(rx) * y4 - sin(rx) * z4;
  float z5 = sin(rx) * y4 + cos(rx) * z4;

  //Rotate around the z axis (rz), then translate
  float x6 = cos(rz) * x5 - sin(rz) * y5 + tx;
  float y6 = sin(rz) * x5 + cos(rz) * y5 + ty;
  float z6 = z5 + tz;

  //Rotate around the z axis after translation（imuRollStart）
  float x7 = cos(imuRollStart) * (x6 - imuShiftFromStartX) 
           - sin(imuRollStart) * (y6 - imuShiftFromStartY);
  float y7 = sin(imuRollStart) * (x6 - imuShiftFromStartX) 
           + cos(imuRollStart) * (y6 - imuShiftFromStartY);
  float z7 = z6 - imuShiftFromStartZ;

  //Rotate around the x axis（imuPitchStart）
  float x8 = x7;
  float y8 = cos(imuPitchStart) * y7 - sin(imuPitchStart) * z7;
  float z8 = sin(imuPitchStart) * y7 + cos(imuPitchStart) * z7;

  //Rotate around the y axis（imuYawStart）
  float x9 = cos(imuYawStart) * x8 + sin(imuYawStart) * z8;
  float y9 = y8;
  float z9 = -sin(imuYawStart) * x8 + cos(imuYawStart) * z8;

  //Rotate around the y axis（-imuYawLast）
  float x10 = cos(imuYawLast) * x9 - sin(imuYawLast) * z9;
  float y10 = y9;
  float z10 = sin(imuYawLast) * x9 + cos(imuYawLast) * z9;

  //Rotate around the x axis（-imuPitchLast）
  float x11 = x10;
  float y11 = cos(imuPitchLast) * y10 + sin(imuPitchLast) * z10;
  float z11 = -sin(imuPitchLast) * y10 + cos(imuPitchLast) * z10;

  //Rotate around the z axis（-imuRollLast）
  po->x = cos(imuRollLast) * x11 + sin(imuRollLast) * y11;
  po->y = -sin(imuRollLast) * x11 + cos(imuRollLast) * y11;
  po->z = z11;
  //Keep only the wire number
  po->intensity = int(pi->intensity);
}

//Use IMU to correct the rotation amount, according to the initial Euler angle, the Euler angle of the current point cloud is corrected // Euler angle이 머임 정확히?
void PluginIMURotation(float bcx, float bcy, float bcz, float blx, float bly, float blz,  // 이 모듈에서 line 983에서 1번만 사용됨
                       float alx, float aly, float alz, float &acx, float &acy, float &acz)
{
  float sbcx = sin(bcx);
  float cbcx = cos(bcx);
  float sbcy = sin(bcy);
  float cbcy = cos(bcy);
  float sbcz = sin(bcz);
  float cbcz = cos(bcz);

  float sblx = sin(blx);
  float cblx = cos(blx);
  float sbly = sin(bly);
  float cbly = cos(bly);
  float sblz = sin(blz);
  float cblz = cos(blz);

  float salx = sin(alx);
  float calx = cos(alx);
  float saly = sin(aly);
  float caly = cos(aly);
  float salz = sin(alz);
  float calz = cos(alz);

  float srx = -sbcx*(salx*sblx + calx*caly*cblx*cbly + calx*cblx*saly*sbly) 
            - cbcx*cbcz*(calx*saly*(cbly*sblz - cblz*sblx*sbly) 
            - calx*caly*(sbly*sblz + cbly*cblz*sblx) + cblx*cblz*salx) 
            - cbcx*sbcz*(calx*caly*(cblz*sbly - cbly*sblx*sblz) 
            - calx*saly*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sblz);
  acx = -asin(srx);

  float srycrx = (cbcy*sbcz - cbcz*sbcx*sbcy)*(calx*saly*(cbly*sblz - cblz*sblx*sbly) 
               - calx*caly*(sbly*sblz + cbly*cblz*sblx) + cblx*cblz*salx) 
               - (cbcy*cbcz + sbcx*sbcy*sbcz)*(calx*caly*(cblz*sbly - cbly*sblx*sblz) 
               - calx*saly*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sblz) 
               + cbcx*sbcy*(salx*sblx + calx*caly*cblx*cbly + calx*cblx*saly*sbly);
  float crycrx = (cbcz*sbcy - cbcy*sbcx*sbcz)*(calx*caly*(cblz*sbly - cbly*sblx*sblz) 
               - calx*saly*(cbly*cblz + sblx*sbly*sblz) + cblx*salx*sblz) 
               - (sbcy*sbcz + cbcy*cbcz*sbcx)*(calx*saly*(cbly*sblz - cblz*sblx*sbly) 
               - calx*caly*(sbly*sblz + cbly*cblz*sblx) + cblx*cblz*salx) 
               + cbcx*cbcy*(salx*sblx + calx*caly*cblx*cbly + calx*cblx*saly*sbly);
  acy = atan2(srycrx / cos(acx), crycrx / cos(acx));
  
  float srzcrx = sbcx*(cblx*cbly*(calz*saly - caly*salx*salz) 
               - cblx*sbly*(caly*calz + salx*saly*salz) + calx*salz*sblx) 
               - cbcx*cbcz*((caly*calz + salx*saly*salz)*(cbly*sblz - cblz*sblx*sbly) 
               + (calz*saly - caly*salx*salz)*(sbly*sblz + cbly*cblz*sblx) 
               - calx*cblx*cblz*salz) + cbcx*sbcz*((caly*calz + salx*saly*salz)*(cbly*cblz 
               + sblx*sbly*sblz) + (calz*saly - caly*salx*salz)*(cblz*sbly - cbly*sblx*sblz) 
               + calx*cblx*salz*sblz);
  float crzcrx = sbcx*(cblx*sbly*(caly*salz - calz*salx*saly) 
               - cblx*cbly*(saly*salz + caly*calz*salx) + calx*calz*sblx) 
               + cbcx*cbcz*((saly*salz + caly*calz*salx)*(sbly*sblz + cbly*cblz*sblx) 
               + (caly*salz - calz*salx*saly)*(cbly*sblz - cblz*sblx*sbly) 
               + calx*calz*cblx*cblz) - cbcx*sbcz*((saly*salz + caly*calz*salx)*(cblz*sbly 
               - cbly*sblx*sblz) + (caly*salz - calz*salx*saly)*(cbly*cblz + sblx*sbly*sblz) 
               - calx*calz*cblx*sblz);
  acz = atan2(srzcrx / cos(acx), crzcrx / cos(acx));
}

//Relative to the first point cloud that is the origin, accumulate the amount of rotation
void AccumulateRotation(float cx, float cy, float cz, float lx, float ly, float lz, 
                        float &ox, float &oy, float &oz)
{
  float srx = cos(lx)*cos(cx)*sin(ly)*sin(cz) - cos(cx)*cos(cz)*sin(lx) - cos(lx)*cos(ly)*sin(cx);
  ox = -asin(srx);

  float srycrx = sin(lx)*(cos(cy)*sin(cz) - cos(cz)*sin(cx)*sin(cy)) + cos(lx)*sin(ly)*(cos(cy)*cos(cz) 
               + sin(cx)*sin(cy)*sin(cz)) + cos(lx)*cos(ly)*cos(cx)*sin(cy);
  float crycrx = cos(lx)*cos(ly)*cos(cx)*cos(cy) - cos(lx)*sin(ly)*(cos(cz)*sin(cy) 
               - cos(cy)*sin(cx)*sin(cz)) - sin(lx)*(sin(cy)*sin(cz) + cos(cy)*cos(cz)*sin(cx));
  oy = atan2(srycrx / cos(ox), crycrx / cos(ox));

  float srzcrx = sin(cx)*(cos(lz)*sin(ly) - cos(ly)*sin(lx)*sin(lz)) + cos(cx)*sin(cz)*(cos(ly)*cos(lz) 
               + sin(lx)*sin(ly)*sin(lz)) + cos(lx)*cos(cx)*cos(cz)*sin(lz);
  float crzcrx = cos(lx)*cos(lz)*cos(cx)*cos(cz) - cos(cx)*sin(cz)*(cos(ly)*sin(lz) 
               - cos(lz)*sin(lx)*sin(ly)) - sin(cx)*(sin(ly)*sin(lz) + cos(ly)*cos(lz)*sin(lx));
  oz = atan2(srzcrx / cos(ox), crzcrx / cos(ox));
}
// 아래 4개의 void f는 다 완전 같은 형식
void laserCloudSharpHandler(const sensor_msgs::PointCloud2ConstPtr& cornerPointsSharp2)
{
  timeCornerPointsSharp = cornerPointsSharp2->header.stamp.toSec();

  cornerPointsSharp->clear(); // clear f 머임?
  pcl::fromROSMsg(*cornerPointsSharp2, *cornerPointsSharp);
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*cornerPointsSharp,*cornerPointsSharp, indices);
  newCornerPointsSharp = true;
}

void laserCloudLessSharpHandler(const sensor_msgs::PointCloud2ConstPtr& cornerPointsLessSharp2)
{
  timeCornerPointsLessSharp = cornerPointsLessSharp2->header.stamp.toSec();

  cornerPointsLessSharp->clear();
  pcl::fromROSMsg(*cornerPointsLessSharp2, *cornerPointsLessSharp);
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*cornerPointsLessSharp,*cornerPointsLessSharp, indices);
  newCornerPointsLessSharp = true;
}

void laserCloudFlatHandler(const sensor_msgs::PointCloud2ConstPtr& surfPointsFlat2)
{
  timeSurfPointsFlat = surfPointsFlat2->header.stamp.toSec();

  surfPointsFlat->clear();
  pcl::fromROSMsg(*surfPointsFlat2, *surfPointsFlat);
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*surfPointsFlat,*surfPointsFlat, indices);
  newSurfPointsFlat = true;
}

void laserCloudLessFlatHandler(const sensor_msgs::PointCloud2ConstPtr& surfPointsLessFlat2)
{
  timeSurfPointsLessFlat = surfPointsLessFlat2->header.stamp.toSec();

  surfPointsLessFlat->clear();
  pcl::fromROSMsg(*surfPointsLessFlat2, *surfPointsLessFlat);
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*surfPointsLessFlat,*surfPointsLessFlat, indices);
  newSurfPointsLessFlat = true;
}

//Receive all points
void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudFullRes2)
{
  timeLaserCloudFullRes = laserCloudFullRes2->header.stamp.toSec();

  laserCloudFullRes->clear();
  pcl::fromROSMsg(*laserCloudFullRes2, *laserCloudFullRes);
  std::vector<int> indices;
  pcl::removeNaNFromPointCloud(*laserCloudFullRes,*laserCloudFullRes, indices);
  newLaserCloudFullRes = true;
}

//Receive imu message
void imuTransHandler(const sensor_msgs::PointCloud2ConstPtr& imuTrans2) // main 함수 line 444에서 사용하네
{
  timeImuTrans = imuTrans2->header.stamp.toSec();

  imuTrans->clear();
  pcl::fromROSMsg(*imuTrans2, *imuTrans);

  //Extract imu information based on the message sent
  imuPitchStart = imuTrans->points[0].x;
  imuYawStart = imuTrans->points[0].y;
  imuRollStart = imuTrans->points[0].z;

  imuPitchLast = imuTrans->points[1].x;
  imuYawLast = imuTrans->points[1].y;
  imuRollLast = imuTrans->points[1].z;

  imuShiftFromStartX = imuTrans->points[2].x;
  imuShiftFromStartY = imuTrans->points[2].y;
  imuShiftFromStartZ = imuTrans->points[2].z;

  imuVeloFromStartX = imuTrans->points[3].x;
  imuVeloFromStartY = imuTrans->points[3].y;
  imuVeloFromStartZ = imuTrans->points[3].z;

  newImuTrans = true;
}


int main(int argc, char** argv)
{
  ros::init(argc, argv, "laserOdometry");
  ros::NodeHandle nh;

  ros::Subscriber subCornerPointsSharp = nh.subscribe<sensor_msgs::PointCloud2>
                                         ("/laser_cloud_sharp", 2, laserCloudSharpHandler);

  ros::Subscriber subCornerPointsLessSharp = nh.subscribe<sensor_msgs::PointCloud2>
                                             ("/laser_cloud_less_sharp", 2, laserCloudLessSharpHandler);

  ros::Subscriber subSurfPointsFlat = nh.subscribe<sensor_msgs::PointCloud2>
                                      ("/laser_cloud_flat", 2, laserCloudFlatHandler);

  ros::Subscriber subSurfPointsLessFlat = nh.subscribe<sensor_msgs::PointCloud2>
                                          ("/laser_cloud_less_flat", 2, laserCloudLessFlatHandler);

  ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2> 
                                         ("/velodyne_cloud_2", 2, laserCloudFullResHandler);

  ros::Subscriber subImuTrans = nh.subscribe<sensor_msgs::PointCloud2> 
                                ("/imu_trans", 5, imuTransHandler);

  ros::Publisher pubLaserCloudCornerLast = nh.advertise<sensor_msgs::PointCloud2>
                                           ("/laser_cloud_corner_last", 2); // 여기 뒤의 숫자 2, 5 이런거 머임?

  ros::Publisher pubLaserCloudSurfLast = nh.advertise<sensor_msgs::PointCloud2>
                                         ("/laser_cloud_surf_last", 2);

  ros::Publisher pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2> 
                                        ("/velodyne_cloud_3", 2);
  
  ros::Publisher pubLaserOdometry = nh.advertise<nav_msgs::Odometry> ("/laser_odom_to_init", 5);
  nav_msgs::Odometry laserOdometry; 
  laserOdometry.header.frame_id = "/camera_init";
  laserOdometry.child_frame_id = "/laser_odom";

  tf::TransformBroadcaster tfBroadcaster;
  tf::StampedTransform laserOdometryTrans;
  laserOdometryTrans.frame_id_ = "/camera_init";
  laserOdometryTrans.child_frame_id_ = "/laser_odom";

  std::vector<int> pointSearchInd;//The order of the searched points
  std::vector<float> pointSearchSqDis;//The squared distance of the searched point

  PointType pointOri, pointSel/*Selected feature points*/, tripod1, tripod2, tripod3/*Corresponding points of feature points*/, pointProj/*unused*/, coeff;

  //Sign of degradation
  bool isDegenerate = false;
  //P matrix, prediction matrix
  cv::Mat matP(6, 6, CV_32F, cv::Scalar::all(0)); // code 보는 중 지금까지 처음으로 cv 등장

  int frameCount = skipFrameNum;
  ros::Rate rate(100);
  bool status = ros::ok(); //ok? print문처럼 여기까지 동작했나 확인하는 코드인듯!? anyway, here ros::ok() = true!
  while (status) {
    ros::spinOnce();
// 아래의 if문은 line 1073에서 끝남 // 걍 이 main f의 유일한 핵심임
    if (newCornerPointsSharp && newCornerPointsLessSharp && newSurfPointsFlat &&  // 여기 조건문들이 odom계산 threshold 인듯?
        newSurfPointsLessFlat && newLaserCloudFullRes && newImuTrans &&
        fabs(timeCornerPointsSharp - timeSurfPointsLessFlat) < 0.005 && // fabs f = 실수 절대값
        fabs(timeCornerPointsLessSharp - timeSurfPointsLessFlat) < 0.005 && // 여기 뺴는 식들에서 다 빼는 항은 timeSurfPointsLessFlat이네. 신기.
        fabs(timeSurfPointsFlat - timeSurfPointsLessFlat) < 0.005 &&
        fabs(timeLaserCloudFullRes - timeSurfPointsLessFlat) < 0.005 && // 지금 여기 이 if 문의 실행문이 1073line까지 이어짐.
        fabs(timeImuTrans - timeSurfPointsLessFlat) < 0.005) {  //Synchronization, to ensure that the feature points and IMU information of the same point cloud are received at the same time before entering
      newCornerPointsSharp = false;
      newCornerPointsLessSharp = false;
      newSurfPointsFlat = false;
      newSurfPointsLessFlat = false;
      newLaserCloudFullRes = false;
      newImuTrans = false;

      //Send the first point cloud data set to laserMapping, and start processing from the next point cloud data 
      if (!systemInited) { // 여기서의 if 문은 just initialization 일 뿐이구만?
        //Exchange cornerPointsLessSharp with laserCloudCornerLast to save the value of cornerPointsLessSharp for next round use
        pcl::PointCloud<PointType>::Ptr laserCloudTemp = cornerPointsLessSharp; // 모든 edge points와 plane points를 담기위한 temporary variable 생성. 이 laserCloudTemp 변수는 1041line에서 또 사용됨.
        cornerPointsLessSharp = laserCloudCornerLast; // laserCloudCornerLast는 86line에 정의되어 있음. = less sharp points of last frame
        laserCloudCornerLast = laserCloudTemp; // 모든 edge points

        //Exchange surfPointLessFlat with laserCloudSurfLast, the purpose is to save the value of surfPointsLessFlat for next round use
        laserCloudTemp = surfPointsLessFlat;
        surfPointsLessFlat = laserCloudSurfLast;
        laserCloudSurfLast = laserCloudTemp; // 모든 plane points
//*********************************************************************************
        //Use the feature points of the previous frame to construct a kd-tree // 여기서 kdtreeCornerLast 등의 변수는 kdtree 라이브러리로 만든거임.
        kdtreeCornerLast->setInputCloud(laserCloudCornerLast);//Collection of all edge points // lessSharpPoints로 만들어졌던데 왜 all 임?
        kdtreeSurfLast->setInputCloud(laserCloudSurfLast);//Collection of all plane points // 

        //Send cornerPointsLessSharp and surfPointLessFlat points, that is, edge points and plane points to laserMapping respectively
        sensor_msgs::PointCloud2 laserCloudCornerLast2; // edge // 변수 선언
        pcl::toROSMsg(*laserCloudCornerLast, laserCloudCornerLast2); // ROSMsg로 변환시키는 코드, 다른 node로 MSG 보내기 위해
        laserCloudCornerLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat); // 이거는 edge, plane 둘다 tiemSurfPointsLessFlat이네
        laserCloudCornerLast2.header.frame_id = "/camera"; // 여기 header 부분들은 MSG의 config file인듯?
        pubLaserCloudCornerLast.publish(laserCloudCornerLast2); // 이게 이제 결론적으로 MSG로 publish해주는 code

        sensor_msgs::PointCloud2 laserCloudSurfLast2; // plane
        pcl::toROSMsg(*laserCloudSurfLast, laserCloudSurfLast2);
        laserCloudSurfLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
        laserCloudSurfLast2.header.frame_id = "/camera";
        pubLaserCloudSurfLast.publish(laserCloudSurfLast2);

        //Remember the roll angle and pitch angle of the origin
        transformSum[0] += imuPitchStart;
        transformSum[2] += imuRollStart;

        systemInited = true;
        continue;
      }

      //The initial value of T translation is assigned to the acceleration and deceleration displacement, which is the direction of the gradient drop (using the last converted T (a sweep constant speed model), and at the same time subtracting the constant movement displacement on the basis, that is, only the acceleration and deceleration are considered Displacement) //ㅇDisplacement = 변위
      transform[3] -= imuVeloFromStartX * scanPeriod; // 거 = 속 x 시
      transform[4] -= imuVeloFromStartY * scanPeriod; // translation은 x,y,z 순으로 3,4,5에 저장되어잇네. rotation과 다르게.
      transform[5] -= imuVeloFromStartZ * scanPeriod; // T에서 translation의 initial value는 odom으로 하는 게 아니라, IMU로 setting.

      if (laserCloudCornerLastNum > 10 && laserCloudSurfLastNum > 100) { // line 960에서 끝나는 if 문임
        std::vector<int> indices;
        pcl::removeNaNFromPointCloud(*cornerPointsSharp,*cornerPointsSharp, indices);
        int cornerPointsSharpNum = cornerPointsSharp->points.size();
        int surfPointsFlatNum = surfPointsFlat->points.size();
        
        //Levenberg-Marquardt algorithm (L-M method), non-linear least squares algorithm, a kind of optimization algorithm
        //Up to 25 iterations
        for (int iterCount = 0; iterCount < 25; iterCount++) { // line 949까지.
          laserCloudOri->clear();
          coeffSel->clear();

          //Process the feature points with the largest curvature in the current point cloud, find two closest distance points from the feature points with relatively large curvature in the last point cloud, one point is searched by kd-tree, and the other is adjacent to it according to the found point Find another point with the closest distance
          for (int i = 0; i < cornerPointsSharpNum; i++) { // 여기부터 edge거 뭔가 계산하는 듯? line 688까지
            TransformToStart(&cornerPointsSharp->points[i], &pointSel); // void f 에서 정의됨

            //Every five iterations, re-find the closest point
            if (iterCount % 5 == 0) {
              std::vector<int> indices;
              pcl::removeNaNFromPointCloud(*laserCloudCornerLast,*laserCloudCornerLast, indices);
              //kd-tree finds a point at the closest distance. The edge points are not filtered by voxel grid. Generally, there are relatively few edge points, so no filtering is done.
              kdtreeCornerLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
              int closestPointInd = -1, minPointInd2 = -1;

              //Find the point with the smallest distance between the adjacent line and the target point
              //Remind again: velodyne is a line of 2 degrees. Adjacent scanID does not mean that the line numbers are adjacent. The degrees of adjacent lines are different by 2 degrees, that is, the scanID of the line number is different by 2.
              if (pointSearchSqDis[0] < 25) {//If the closest point found is indeed very close
                closestPointInd = pointSearchInd[0];
                //Extract the nearest point line number
                int closestPointScan = int(laserCloudCornerLast->points[closestPointInd].intensity);

                float pointSqDis, minPointSqDis2 = 25;//The initial threshold value is 5 meters, which can roughly filter out the adjacent scanID, but the actual line is not adjacent
                //Find the point with the smallest sum of squares of the closest distance to the target point
                for (int j = closestPointInd + 1; j < cornerPointsSharpNum; j++) {//Search in the direction of increasing scanID
                  if (int(laserCloudCornerLast->points[j].intensity) > closestPointScan + 2.5) {//Non-adjacent lines
                    break;
                  }

                  pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) * 
                               (laserCloudCornerLast->points[j].x - pointSel.x) + 
                               (laserCloudCornerLast->points[j].y - pointSel.y) * 
                               (laserCloudCornerLast->points[j].y - pointSel.y) + 
                               (laserCloudCornerLast->points[j].z - pointSel.z) * 
                               (laserCloudCornerLast->points[j].z - pointSel.z);

                  if (int(laserCloudCornerLast->points[j].intensity) > closestPointScan) {//Make sure that the two points are not on the same scan (adjacent line search should be done with scanID == closestPointScan +/- 1）
                    if (pointSqDis < minPointSqDis2) {//The distance is closer, less than the initial value of 5 meters
                        //Update minimum distance and point order
                      minPointSqDis2 = pointSqDis;
                      minPointInd2 = j;
                    }
                  }
                }

                //The same
                for (int j = closestPointInd - 1; j >= 0; j--) {//Search in the direction of decreasing scanID
                  if (int(laserCloudCornerLast->points[j].intensity) < closestPointScan - 2.5) {
                    break;
                  }

                  pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) * 
                               (laserCloudCornerLast->points[j].x - pointSel.x) + 
                               (laserCloudCornerLast->points[j].y - pointSel.y) * 
                               (laserCloudCornerLast->points[j].y - pointSel.y) + 
                               (laserCloudCornerLast->points[j].z - pointSel.z) * 
                               (laserCloudCornerLast->points[j].z - pointSel.z);

                  if (int(laserCloudCornerLast->points[j].intensity) < closestPointScan) {
                    if (pointSqDis < minPointSqDis2) {
                      minPointSqDis2 = pointSqDis;
                      minPointInd2 = j;
                    }
                  }
                }
              }

              //Remember the order of the points that make up the line
              pointSearchCornerInd1[i] = closestPointInd;//kd-tree closest distance point, -1 means no point is found to be satisfied
              pointSearchCornerInd2[i] = minPointInd2;//The other nearest one, -1 means that no satisfying point was found
            }

            if (pointSearchCornerInd2[i] >= 0) {//Greater than or equal to 0, not equal to -1, indicating that both points have been found
              tripod1 = laserCloudCornerLast->points[pointSearchCornerInd1[i]];
              tripod2 = laserCloudCornerLast->points[pointSearchCornerInd2[i]];

              //The selected feature point is denoted as O, the closest distance point of kd-tree is denoted as A, and the other closest distance point is denoted as B选择的特征点记为O，kd-tree最近距离点记为A，另一个最近距离点记为B
              float x0 = pointSel.x;
              float y0 = pointSel.y;
              float z0 = pointSel.z;
              float x1 = tripod1.x;
              float y1 = tripod1.y;
              float z1 = tripod1.z;
              float x2 = tripod2.x;
              float y2 = tripod2.y;
              float z2 = tripod2.z;

              //Vector OA = (x0-x1, y0-y1, z0-z1), vector OB = (x0-x2, y0-y2, z0-z2), vector AB = (x1-x2, y1-y2, z1-z2)
              //The vector product (ie, cross product) of the vector OA OB is:
              //|  i      j      k  |
              //|x0-x1  y0-y1  z0-z1|
              //|x0-x2  y0-y2  z0-z2|
              //Modulo：
              float a012 = sqrt(((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
                         * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                         + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
                         * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                         + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))
                         * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));

              //The distance between the two closest points, that is, the modulus of the vector AB
              float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));

              //The component of the vector product of the unit vector in the AB direction and the unit normal vector of the OAB plane on each axis (direction of d)
              //x-axis component i
              float la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                       + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012 / l12;

              //y-axis component j
              float lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1)) 
                       - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

              //z-axis component k
              float lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1)) 
                       + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;

              //The distance from point to line, d = |vector OA cross product vector OB|/|AB|
              float ld2 = a012 / l12;

              //unused
              pointProj = pointSel;
              pointProj.x -= la * ld2;
              pointProj.y -= lb * ld2;
              pointProj.z -= lc * ld2;

              //Weight calculation, the greater the distance, the smaller the weight, the smaller the distance, the greater the weight, the weight range obtained is <=1
              float s = 1;
              if (iterCount >= 5) {//Start to increase the weight factor after 5 iterations
                s = 1 - 1.8 * fabs(ld2);
              }

              //Consider the weight
              coeff.x = s * la;
              coeff.y = s * lb;
              coeff.z = s * lc;
              coeff.intensity = s * ld2;

              if (s > 0.1 && ld2 != 0) {//Only keep the points with significant weight, that is, the points with a relatively small distance, and discard the points with zero distance
                laserCloudOri->push_back(cornerPointsSharp->points[i]);
                coeffSel->push_back(coeff);
              }
            }
          } // line 550부터 시작

          //For the point with the smallest curvature received this time, find three points from the point with relatively small curvature of the point cloud received last time to form a plane. Three find on different lines to meet the requirements
          for (int i = 0; i < surfPointsFlatNum; i++) { // 여기부터 plane거 뭔가 계산하는 듯? line 817까지
            TransformToStart(&surfPointsFlat->points[i], &pointSel); // 위의 void f 에서 정의됨

            if (iterCount % 5 == 0) {
                //kd-tree nearest point search, search in the plane points after voxel grid filtering, generally there are too many plane points, and the amount of data to find the nearest point after filtering is small
              kdtreeSurfLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
              int closestPointInd = -1, minPointInd2 = -1, minPointInd3 = -1;
              if (pointSearchSqDis[0] < 25) {
                closestPointInd = pointSearchInd[0];
                int closestPointScan = int(laserCloudSurfLast->points[closestPointInd].intensity);

                float pointSqDis, minPointSqDis2 = 25, minPointSqDis3 = 25;
                for (int j = closestPointInd + 1; j < surfPointsFlatNum; j++) {
                  if (int(laserCloudSurfLast->points[j].intensity) > closestPointScan + 2.5) {
                    break;
                  }

                  pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) * 
                               (laserCloudSurfLast->points[j].x - pointSel.x) + 
                               (laserCloudSurfLast->points[j].y - pointSel.y) * 
                               (laserCloudSurfLast->points[j].y - pointSel.y) + 
                               (laserCloudSurfLast->points[j].z - pointSel.z) * 
                               (laserCloudSurfLast->points[j].z - pointSel.z);

                  if (int(laserCloudSurfLast->points[j].intensity) <= closestPointScan) {//If the line number of the point is less than or equal to the line number of the nearest point (it should be equal at most, that is, points on the same line)
                     if (pointSqDis < minPointSqDis2) {
                       minPointSqDis2 = pointSqDis;
                       minPointInd2 = j;
                     }
                  } else {//If the point is on a line greater than that
                     if (pointSqDis < minPointSqDis3) {
                       minPointSqDis3 = pointSqDis;
                       minPointInd3 = j;
                     }
                  }
                }


                //The same
                for (int j = closestPointInd - 1; j >= 0; j--) {
                  if (int(laserCloudSurfLast->points[j].intensity) < closestPointScan - 2.5) {
                    break;
                  }

                  pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) * 
                               (laserCloudSurfLast->points[j].x - pointSel.x) + 
                               (laserCloudSurfLast->points[j].y - pointSel.y) * 
                               (laserCloudSurfLast->points[j].y - pointSel.y) + 
                               (laserCloudSurfLast->points[j].z - pointSel.z) * 
                               (laserCloudSurfLast->points[j].z - pointSel.z);

                  if (int(laserCloudSurfLast->points[j].intensity) >= closestPointScan) {
                    if (pointSqDis < minPointSqDis2) {
                      minPointSqDis2 = pointSqDis;
                      minPointInd2 = j;
                    }
                  } else {
                    if (pointSqDis < minPointSqDis3) {
                      minPointSqDis3 = pointSqDis;
                      minPointInd3 = j;
                    }
                  }
                }
              }

              pointSearchSurfInd1[i] = closestPointInd;//kd-tree closest distance point, -1 means no point that meets the requirements is found
              pointSearchSurfInd2[i] = minPointInd2;//The closest point on the same line number, -1 means no point that meets the requirements is found
              pointSearchSurfInd3[i] = minPointInd3;//The closest point on different line numbers, -1 means no point that meets the requirements is found
            }

            if (pointSearchSurfInd2[i] >= 0 && pointSearchSurfInd3[i] >= 0) {//Found three points
              tripod1 = laserCloudSurfLast->points[pointSearchSurfInd1[i]];//Point a
              tripod2 = laserCloudSurfLast->points[pointSearchSurfInd2[i]];//Point b
              tripod3 = laserCloudSurfLast->points[pointSearchSurfInd3[i]];//Point c

              //vectorAB = (tripod2.x - tripod1.x, tripod2.y - tripod1.y, tripod2.z - tripod1.z)
              //vectorAC = (tripod3.x - tripod1.x, tripod3.y - tripod1.y, tripod3.z - tripod1.z)

              //vectorAB AC's vector product (ie cross product), get the normal vector
              //x-axis component vector i
              float pa = (tripod2.y - tripod1.y) * (tripod3.z - tripod1.z) 
                       - (tripod3.y - tripod1.y) * (tripod2.z - tripod1.z);
              //y-axis component vector j
              float pb = (tripod2.z - tripod1.z) * (tripod3.x - tripod1.x) 
                       - (tripod3.z - tripod1.z) * (tripod2.x - tripod1.x);
              //z-axis component vector k
              float pc = (tripod2.x - tripod1.x) * (tripod3.y - tripod1.y) 
                       - (tripod3.x - tripod1.x) * (tripod2.y - tripod1.y);
              float pd = -(pa * tripod1.x + pb * tripod1.y + pc * tripod1.z);

              //The norm of the normal vector
              float ps = sqrt(pa * pa + pb * pb + pc * pc);
              //pa pb pc is the unit vector of the normal vector in all directions
              pa /= ps;
              pb /= ps;
              pc /= ps;
              pd /= ps;

              //The distance from the point to the surface: the dot product of the vector OA and the normal vector divided by the modulus of the normal vector
              float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

              //unused
              pointProj = pointSel;
              pointProj.x -= pa * pd2;
              pointProj.y -= pb * pd2;
              pointProj.z -= pc * pd2;

              //Calculate the weight in the same way
              float s = 1;
              if (iterCount >= 5) {
                s = 1 - 1.8 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x
                  + pointSel.y * pointSel.y + pointSel.z * pointSel.z));
              }

              //Consider the weight
              coeff.x = s * pa;
              coeff.y = s * pb;
              coeff.z = s * pc;
              coeff.intensity = s * pd2;

              if (s > 0.1 && pd2 != 0) {
                  //Save the original point and the corresponding coefficient
                laserCloudOri->push_back(surfPointsFlat->points[i]);
                coeffSel->push_back(coeff);
              }
            }
          } // line 691부터 시작

          int pointSelNum = laserCloudOri->points.size();
          //At least 10 feature points that meet the requirements, and the number of feature matching is too small. Discard this frame of data
          if (pointSelNum < 10) {
            continue;
          } // line 545 에서의 for 문을 그만 하란 if 문임

          cv::Mat matA(pointSelNum, 6, CV_32F, cv::Scalar::all(0));
          cv::Mat matAt(6, pointSelNum, CV_32F, cv::Scalar::all(0));
          cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
          cv::Mat matB(pointSelNum, 1, CV_32F, cv::Scalar::all(0));
          cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
          cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));

          //Calculate matA, matB matrix
          for (int i = 0; i < pointSelNum; i++) { // 892
            pointOri = laserCloudOri->points[i];
            coeff = coeffSel->points[i];

            float s = 1;

            float srx = sin(s * transform[0]);
            float crx = cos(s * transform[0]);
            float sry = sin(s * transform[1]);
            float cry = cos(s * transform[1]);
            float srz = sin(s * transform[2]);
            float crz = cos(s * transform[2]);
            float tx = s * transform[3];
            float ty = s * transform[4];
            float tz = s * transform[5];
// ppt 각각의 그림들에 설명 붙여서 업로드(준호) + pcl_viewer 어떻게 하는지 flow 제대로 정리해서 올리기(창원)
// 가제보 이외의 ros에서 돌릴 수 있는 유사 시뮬레이션 환경 알아봐서 알려주기. 카라라고 있음 자율주행에서 쓰는 거임. 허스키봇 여러가지 환경에서 쓰는 거 알아볼 수 있을거임(종욱)
// ros에서 구현된 디텍션 알고리즘이랑 슬램 을 섞어서 예전에 발표할때 종욱이 올렸던 거임.
// 슬램 매핑해서 맵 만들고 거기에다가 디텍션까지 우리가 한번에 다 해주는 게 교수님 목표임
            float arx = (-s*crx*sry*srz*pointOri.x + s*crx*crz*sry*pointOri.y + s*srx*sry*pointOri.z 
                      + s*tx*crx*sry*srz - s*ty*crx*crz*sry - s*tz*srx*sry) * coeff.x
                      + (s*srx*srz*pointOri.x - s*crz*srx*pointOri.y + s*crx*pointOri.z
                      + s*ty*crz*srx - s*tz*crx - s*tx*srx*srz) * coeff.y
                      + (s*crx*cry*srz*pointOri.x - s*crx*cry*crz*pointOri.y - s*cry*srx*pointOri.z
                      + s*tz*cry*srx + s*ty*crx*cry*crz - s*tx*crx*cry*srz) * coeff.z;

            float ary = ((-s*crz*sry - s*cry*srx*srz)*pointOri.x 
                      + (s*cry*crz*srx - s*sry*srz)*pointOri.y - s*crx*cry*pointOri.z 
                      + tx*(s*crz*sry + s*cry*srx*srz) + ty*(s*sry*srz - s*cry*crz*srx) 
                      + s*tz*crx*cry) * coeff.x
                      + ((s*cry*crz - s*srx*sry*srz)*pointOri.x 
                      + (s*cry*srz + s*crz*srx*sry)*pointOri.y - s*crx*sry*pointOri.z
                      + s*tz*crx*sry - ty*(s*cry*srz + s*crz*srx*sry) 
                      - tx*(s*cry*crz - s*srx*sry*srz)) * coeff.z;

            float arz = ((-s*cry*srz - s*crz*srx*sry)*pointOri.x + (s*cry*crz - s*srx*sry*srz)*pointOri.y
                      + tx*(s*cry*srz + s*crz*srx*sry) - ty*(s*cry*crz - s*srx*sry*srz)) * coeff.x
                      + (-s*crx*crz*pointOri.x - s*crx*srz*pointOri.y
                      + s*ty*crx*srz + s*tx*crx*crz) * coeff.y
                      + ((s*cry*crz*srx - s*sry*srz)*pointOri.x + (s*crz*sry + s*cry*srx*srz)*pointOri.y
                      + tx*(s*sry*srz - s*cry*crz*srx) - ty*(s*crz*sry + s*cry*srx*srz)) * coeff.z;

            float atx = -s*(cry*crz - srx*sry*srz) * coeff.x + s*crx*srz * coeff.y 
                      - s*(crz*sry + cry*srx*srz) * coeff.z;
  
            float aty = -s*(cry*srz + crz*srx*sry) * coeff.x - s*crx*crz * coeff.y 
                      - s*(sry*srz - cry*crz*srx) * coeff.z;
  
            float atz = s*crx*sry * coeff.x - s*srx * coeff.y - s*crx*cry * coeff.z;

            float d2 = coeff.intensity;

            matA.at<float>(i, 0) = arx;
            matA.at<float>(i, 1) = ary;
            matA.at<float>(i, 2) = arz;
            matA.at<float>(i, 3) = atx;
            matA.at<float>(i, 4) = aty;
            matA.at<float>(i, 5) = atz;
            matB.at<float>(i, 0) = -0.05 * d2;
          } // 833
          cv::transpose(matA, matAt);
          matAtA = matAt * matA;
          matAtB = matAt * matB;
          //Solve for matAtA * matX = matAtB
          cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

          if (iterCount == 0) {
            //Eigenvalue 1*6 matrix
            cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
            //Eigenvector 6*6 matrix
            cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
            cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

            //Solve for eigenvalues/eigenvectors
            cv::eigen(matAtA, matE, matV);
            matV.copyTo(matV2);

            isDegenerate = false;
            //Eigenvalue threshold
            float eignThre[6] = {10, 10, 10, 10, 10, 10};
            for (int i = 5; i >= 0; i--) {//Search from small to large
              if (matE.at<float>(0, i) < eignThre[i]) {//If the eigenvalue is too small, it is considered to be in a merger environment and degradation has occurred
                for (int j = 0; j < 6; j++) {//The corresponding feature vector is set to 0
                  matV2.at<float>(i, j) = 0;
                }
                isDegenerate = true;
              } else {
                break;
              }
            }

            //Calculate the P matrix
            matP = matV.inv() * matV2;
          }

          if (isDegenerate) {//If degradation occurs, only use the prediction matrix P to calculate
            cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
            matX.copyTo(matX2);
            matX = matP * matX2;
          }

          //Accumulate the amount of rotation and translation for each iteration
          transform[0] += matX.at<float>(0, 0);
          transform[1] += matX.at<float>(1, 0);
          transform[2] += matX.at<float>(2, 0);
          transform[3] += matX.at<float>(3, 0);
          transform[4] += matX.at<float>(4, 0);
          transform[5] += matX.at<float>(5, 0);

          for(int i=0; i<6; i++){
            if(isnan(transform[i]))//Determine if it is not a number
              transform[i]=0;
          }
          //Calculate the amount of rotation and translation, if it is small, stop the iteration
          float deltaR = sqrt(
                              pow(rad2deg(matX.at<float>(0, 0)), 2) +
                              pow(rad2deg(matX.at<float>(1, 0)), 2) +
                              pow(rad2deg(matX.at<float>(2, 0)), 2));
          float deltaT = sqrt(
                              pow(matX.at<float>(3, 0) * 100, 2) +
                              pow(matX.at<float>(4, 0) * 100, 2) +
                              pow(matX.at<float>(5, 0) * 100, 2));

          if (deltaR < 0.1 && deltaT < 0.1) {//Iteration termination condition
            break;
          }
        } // line 545 부터의 for 문
      } // line 537에서 시작하는 if 문

      float rx, ry, rz, tx, ty, tz;
      //What is the amount of rotation relative to the origin, 1.05 times correction in the vertical direction?
      AccumulateRotation(transformSum[0], transformSum[1], transformSum[2],  // 0 = pitch , 1 = yaw, 2 = roll
                         -transform[0], -transform[1] * 1.05, -transform[2], rx, ry, rz);

      float x1 = cos(rz) * (transform[3] - imuShiftFromStartX) 
               - sin(rz) * (transform[4] - imuShiftFromStartY);
      float y1 = sin(rz) * (transform[3] - imuShiftFromStartX) 
               + cos(rz) * (transform[4] - imuShiftFromStartY);
      float z1 = transform[5] * 1.05 - imuShiftFromStartZ;

      float x2 = x1;
      float y2 = cos(rx) * y1 - sin(rx) * z1;
      float z2 = sin(rx) * y1 + cos(rx) * z1;

      //Find the amount of translation relative to the origin // 논문에서와는 다르게 rotation 이 0,1,2 th element이고, translation이 3,4,5 th element이네!
      tx = transformSum[3] - (cos(ry) * x2 + sin(ry) * z2);
      ty = transformSum[4] - y2;
      tz = transformSum[5] - (-sin(ry) * x2 + cos(ry) * z2);

      //Correct the amount of rotation based on the IMU
      PluginIMURotation(rx, ry, rz, imuPitchStart, imuYawStart, imuRollStart, 
                        imuPitchLast, imuYawLast, imuRollLast, rx, ry, rz);

      //Get the transfer matrix in the world coordinate system // 따지고 보면 이게 Worldview에서 accumulated Transform인듯!
      transformSum[0] = rx;
      transformSum[1] = ry;
      transformSum[2] = rz;
      transformSum[3] = tx;
      transformSum[4] = ty;
      transformSum[5] = tz;

      //Euler angles converted to quaternions
      geometry_msgs::Quaternion geoQuat = tf::createQuaternionMsgFromRollPitchYaw(rz, -rx, -ry);

      //publish quaternion and translation amount
      laserOdometry.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
      laserOdometry.pose.pose.orientation.x = -geoQuat.y;
      laserOdometry.pose.pose.orientation.y = -geoQuat.z;
      laserOdometry.pose.pose.orientation.z = geoQuat.x;
      laserOdometry.pose.pose.orientation.w = geoQuat.w;
      laserOdometry.pose.pose.position.x = tx;
      laserOdometry.pose.pose.position.y = ty;
      laserOdometry.pose.pose.position.z = tz;
      pubLaserOdometry.publish(laserOdometry);

      //Broadcast the new coordinate system after translation and rotation(rviz)
      laserOdometryTrans.stamp_ = ros::Time().fromSec(timeSurfPointsLessFlat);
      laserOdometryTrans.setRotation(tf::Quaternion(-geoQuat.y, -geoQuat.z, geoQuat.x, geoQuat.w));
      laserOdometryTrans.setOrigin(tf::Vector3(tx, ty, tz));
      tfBroadcaster.sendTransform(laserOdometryTrans);

      //Projection of points with relatively large and relatively small curvature of the point cloud to the end of the scan
      int cornerPointsLessSharpNum = cornerPointsLessSharp->points.size();
      for (int i = 0; i < cornerPointsLessSharpNum; i++) {
        TransformToEnd(&cornerPointsLessSharp->points[i], &cornerPointsLessSharp->points[i]); // line 173에서 정의됨
      }

      int surfPointsLessFlatNum = surfPointsLessFlat->points.size();
      for (int i = 0; i < surfPointsLessFlatNum; i++) {
        TransformToEnd(&surfPointsLessFlat->points[i], &surfPointsLessFlat->points[i]);
      }

      frameCount++;
      //All points of the point cloud, each interval of point cloud data is relative to the last point of the point cloud for distortion correction
      if (frameCount >= skipFrameNum + 1) {
        int laserCloudFullResNum = laserCloudFullRes->points.size();
        for (int i = 0; i < laserCloudFullResNum; i++) {
          TransformToEnd(&laserCloudFullRes->points[i], &laserCloudFullRes->points[i]);
        }
      }

      //The point after distortion correction is saved as the last point and the next point cloud comes in for matching
      pcl::PointCloud<PointType>::Ptr laserCloudTemp = cornerPointsLessSharp;
      cornerPointsLessSharp = laserCloudCornerLast;
      laserCloudCornerLast = laserCloudTemp;

      laserCloudTemp = surfPointsLessFlat;
      surfPointsLessFlat = laserCloudSurfLast;
      laserCloudSurfLast = laserCloudTemp;

      laserCloudCornerLastNum = laserCloudCornerLast->points.size();
      laserCloudSurfLastNum = laserCloudSurfLast->points.size();
      //If there are enough points, construct the kd-tree, otherwise discard this frame and use the kd-tree of the previous frame of data
      if (laserCloudCornerLastNum > 10 && laserCloudSurfLastNum > 100) {
        kdtreeCornerLast->setInputCloud(laserCloudCornerLast);
        kdtreeSurfLast->setInputCloud(laserCloudSurfLast);
      }

      //According to the number of frame jumps, publich edge points, plane points and all points are given to laserMapping (send every other frame)
      if (frameCount >= skipFrameNum + 1) {
        frameCount = 0;

        sensor_msgs::PointCloud2 laserCloudCornerLast2;
        pcl::toROSMsg(*laserCloudCornerLast, laserCloudCornerLast2);
        laserCloudCornerLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
        laserCloudCornerLast2.header.frame_id = "/camera";
        pubLaserCloudCornerLast.publish(laserCloudCornerLast2);

        sensor_msgs::PointCloud2 laserCloudSurfLast2;
        pcl::toROSMsg(*laserCloudSurfLast, laserCloudSurfLast2);
        laserCloudSurfLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
        laserCloudSurfLast2.header.frame_id = "/camera";
        pubLaserCloudSurfLast.publish(laserCloudSurfLast2);

        sensor_msgs::PointCloud2 laserCloudFullRes3;
        pcl::toROSMsg(*laserCloudFullRes, laserCloudFullRes3);
        laserCloudFullRes3.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
        laserCloudFullRes3.header.frame_id = "/camera";
        pubLaserCloudFullRes.publish(laserCloudFullRes3);
      }
    } // 이 if 문은 line 481에서 시작

    status = ros::ok();
    rate.sleep(); // 뭔 의미?
  } // 여기까지 while문

  return 0;
} // 여기서 main 함수 끝
