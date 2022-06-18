/******************************읽기전에*****************************************/
/*imu는 x축이 앞으로, y축이 왼쪽, z축이 위쪽인 오른손 좌표계입니다.
  벨로다인 라이더는 x축이 앞으로, y축이 왼쪽, z축이 위로 향하는 오른손 좌표계로 설치되며,
  ScanRegistration은 z축을 앞으로, x축을 왼쪽으로, y축을 위로 하여 좌표축을 오른쪽 좌표계로 교환하여 둘을 통합합니다.
  , 이것은 J. Zhang의 논문에서 사용된 좌표계입니다.
  교환 후: R = Ry(요, 위)*Rx(피치)*Rz(롤, 앞)
*******************************************************************************/

#include <cmath>
#include <vector>

#include <loam_velodyne/common.h>
#include <opencv/cv.h>
#include <nav_msgs/Odometry.h> // 이건 기본적으로 ROS에 있는 headere file?

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h> // kdtree는 pcl library에 기본 장착 됐나 보군!
#include <ros/ros.h>
#include <sensor_msgs/Imu.h> // sensor_msgs library도 기본적으로 있는 건지...? 그럼 Imu도 기본적으로 있는 거일듯!
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h> // 대체 tf는 머지? ROS 책에 있을듯? 책 봐서 확실히 알자
#include <tf/transform_broadcaster.h> // line 900 쯤 보니까 tf도 namespace이네. library라고 보면 되는듯

using std::sin;
using std::cos;
using std::atan2;

//0.1s스캔 주기, 벨로다인 주파수 10Hz, 주기 0.1s
const double scanPeriod = 0.1;

//제어변수초기화
const int systemDelay = 20;//초기 데이터의 처음 20개 프레임은 삭제됩니다.
int systemInitCount = 0;
bool systemInited = false;

//라이더 라인 번호 whats the meaning of this?
const int N_SCANS = 16;

//포인트 클라우드 곡률, 40000은 포인트 클라우드 프레임의 최대 포인트 수입니다.
float cloudCurvature[40000]; // know exatly the meaning of the curvature
//곡률 점에 해당하는 숫자
int cloudSortInd[40000];
//점이 필터링되었는지 여부 플래그: 0-필터링되지 않음, 1-필터링됨
int cloudNeighborPicked[40000];
//포인트 분류 레이블: 2- 곡률이 매우 큰 것을 나타냅니다. 1- 곡률이 상대적으로 큰 것을 나타냅니다. -1- 곡률이 작은 것을 나타냅니다. 0- 곡률이 상대적으로 작습니다(여기서 1은 2를 포함하고 0은 1,0을 포함하고 0은 1,0을 포함하고 1은 포인트 클라우드를 구성합니다. 모든 포인트)
int cloudLabel[40000];

//imu 타임스탬프가 현재 포인트 클라우드 타임스탬프보다 큰 위치
int imuPointerFront = 0; //IMU도 제대로 이해하자
//배열에서 imu가 수신한 가장 최근 지점의 위치
int imuPointerLast = -1;
//IMU 순환 대기열 길이
const int imuQueLength = 200;

//포인트 클라우드 데이터의 첫 번째 포인트의 변위/속도/오일러 각도
float imuRollStart = 0, imuPitchStart = 0, imuYawStart = 0; // 오일러각도 처음
float imuRollCur = 0, imuPitchCur = 0, imuYawCur = 0; // 오일러각도 현재

float imuVeloXStart = 0, imuVeloYStart = 0, imuVeloZStart = 0; // 속도
float imuShiftXStart = 0, imuShiftYStart = 0, imuShiftZStart = 0; // 변위, 이동 = displacement

//Speed ​​and displacement information of the current point
float imuVeloXCur = 0, imuVeloYCur = 0, imuVeloZCur = 0; // current speed
float imuShiftXCur = 0, imuShiftYCur = 0, imuShiftZCur = 0; // current displacement

//The distortion displacement of the current point relative to the first point of the point cloud data each time, and the speed
float imuShiftFromStartXCur = 0, imuShiftFromStartYCur = 0, imuShiftFromStartZCur = 0; // shift = distortion displacement
float imuVeloFromStartXCur = 0, imuVeloFromStartYCur = 0, imuVeloFromStartZCur = 0; // speed

//IMU information
double imuTime[imuQueLength] = {0};
float imuRoll[imuQueLength] = {0};
float imuPitch[imuQueLength] = {0};
float imuYaw[imuQueLength] = {0};

float imuAccX[imuQueLength] = {0};
float imuAccY[imuQueLength] = {0};
float imuAccZ[imuQueLength] = {0};

float imuVeloX[imuQueLength] = {0};
float imuVeloY[imuQueLength] = {0};
float imuVeloZ[imuQueLength] = {0};

float imuShiftX[imuQueLength] = {0};
float imuShiftY[imuQueLength] = {0};
float imuShiftZ[imuQueLength] = {0};

ros::Publisher pubLaserCloud; // pubLaserCloud 같은 건 사용자 정의인듯.
ros::Publisher pubCornerPointsSharp;
ros::Publisher pubCornerPointsLessSharp;
ros::Publisher pubSurfPointsFlat;
ros::Publisher pubSurfPointsLessFlat;
ros::Publisher pubImuTrans;

// void 함수 6개 + main 함수 1개

//Calculate the displacement distortion caused by acceleration and deceleration movement of the point in the point cloud in the local coordinate system relative to the first starting point
void ShiftToStartIMU(float pointTime)
{
  //Calculate the distortion displacement due to acceleration and deceleration relative to the first point (distortion displacement delta_Tg in the global coordinate system)
  //imuShiftFromStartCur = imuShiftCur - (imuShiftStart + imuVeloStart * pointTime)
  imuShiftFromStartXCur = imuShiftXCur - imuShiftXStart - imuVeloXStart * pointTime; // imuShiftFromStartXCur가 delta_Tg인듯!
  imuShiftFromStartYCur = imuShiftYCur - imuShiftYStart - imuVeloYStart * pointTime;
  imuShiftFromStartZCur = imuShiftZCur - imuShiftZStart - imuVeloZStart * pointTime;

  /********************************************************************************
  Rz(roll).inverse * Rx(pitch).inverse * Ry(yaw).inverse * delta_Tg //where is delta_Tg? 
  transfrom from the global frame to the local frame
  *********************************************************************************/

  //Rotate around the y axis (-imuYawStart), namely Ry(yaw).inverse
  float x1 = cos(imuYawStart) * imuShiftFromStartXCur - sin(imuYawStart) * imuShiftFromStartZCur; // 처음엔 cos=1, sin=0이라 x1=imuShiftFromStartXCur임.
  float y1 = imuShiftFromStartYCur;
  float z1 = sin(imuYawStart) * imuShiftFromStartXCur + cos(imuYawStart) * imuShiftFromStartZCur;  // 이것도 cos=1, sin=0이라 z1=imuShiftFromStartZCur임.

  //Rotate around the x axis (-imuPitchStart), namely Rx(pitch).inverse
  float x2 = x1;
  float y2 = cos(imuPitchStart) * y1 + sin(imuPitchStart) * z1;
  float z2 = -sin(imuPitchStart) * y1 + cos(imuPitchStart) * z1;

  //Rotate around the z axis (-imuRollStart), namely Rz(pitch).inverse
  imuShiftFromStartXCur = cos(imuRollStart) * x2 + sin(imuRollStart) * y2;
  imuShiftFromStartYCur = -sin(imuRollStart) * x2 + cos(imuRollStart) * y2;
  imuShiftFromStartZCur = z2;
}

//Calculate the speed distortion (increment) of the point in the point cloud relative to the first starting point in the local coordinate system due to acceleration and deceleration
void VeloToStartIMU()
{
  //Calculate the distortion speed due to acceleration and deceleration relative to the first point (distortion speed increment delta_Vg in the global coordinate system)
  imuVeloFromStartXCur = imuVeloXCur - imuVeloXStart; // 처음엔 일단 0
  imuVeloFromStartYCur = imuVeloYCur - imuVeloYStart;
  imuVeloFromStartZCur = imuVeloZCur - imuVeloZStart;

  /********************************************************************************
    Rz(roll).inverse * Rx(pitch).inverse * Ry(yaw).inverse * delta_Vg
    transfrom from the global frame to the local frame
  *********************************************************************************/
  
  //Rotate around the y axis (-imuYawStart), namely Ry(yaw).inverse
  float x1 = cos(imuYawStart) * imuVeloFromStartXCur - sin(imuYawStart) * imuVeloFromStartZCur;
  float y1 = imuVeloFromStartYCur;
  float z1 = sin(imuYawStart) * imuVeloFromStartXCur + cos(imuYawStart) * imuVeloFromStartZCur;

  //Rotate around the x axis (-imuPitchStart), namely Rx(pitch).inverse
  float x2 = x1;
  float y2 = cos(imuPitchStart) * y1 + sin(imuPitchStart) * z1;
  float z2 = -sin(imuPitchStart) * y1 + cos(imuPitchStart) * z1;

  //Rotate around the z axis (-imuRollStart), namely Rz(pitch).inverse
  imuVeloFromStartXCur = cos(imuRollStart) * x2 + sin(imuRollStart) * y2;
  imuVeloFromStartYCur = -sin(imuRollStart) * x2 + cos(imuRollStart) * y2;
  imuVeloFromStartZCur = z2;
}

//Remove the displacement distortion caused by point cloud acceleration and deceleration
void TransformToStartIMU(PointType *p)
{
  /********************************************************************************
    Ry*Rx*Rz*Pl, transform point to the global frame   /// Pl??
  *********************************************************************************/
  //Rotate around the z axis (imuRollCur)
  float x1 = cos(imuRollCur) * p->x - sin(imuRollCur) * p->y;
  float y1 = sin(imuRollCur) * p->x + cos(imuRollCur) * p->y;
  float z1 = p->z;

  //Rotate around the x axis (imuPitchCur)
  float x2 = x1;
  float y2 = cos(imuPitchCur) * y1 - sin(imuPitchCur) * z1;
  float z2 = sin(imuPitchCur) * y1 + cos(imuPitchCur) * z1;

  //Rotate around the y axis (imuYawCur)
  float x3 = cos(imuYawCur) * x2 + sin(imuYawCur) * z2;
  float y3 = y2;
  float z3 = -sin(imuYawCur) * x2 + cos(imuYawCur) * z2;

  /********************************************************************************
    Rz(roll).inverse * Rx(pitch).inverse * Ry(yaw).inverse * Pg
    transfrom global points to the local frame
  *********************************************************************************/
  
  //Rotate around the y axis (-imuYawStart)
  float x4 = cos(imuYawStart) * x3 - sin(imuYawStart) * z3;
  float y4 = y3;
  float z4 = sin(imuYawStart) * x3 + cos(imuYawStart) * z3;

  //Rotate around the x axis (-imuPitchStart)
  float x5 = x4;
  float y5 = cos(imuPitchStart) * y4 + sin(imuPitchStart) * z4;
  float z5 = -sin(imuPitchStart) * y4 + cos(imuPitchStart) * z4;

  //Rotate around the z axis (-imuRollStart), and then superimpose the translation amount
  p->x = cos(imuRollStart) * x5 + sin(imuRollStart) * y5 + imuShiftFromStartXCur;
  p->y = -sin(imuRollStart) * x5 + cos(imuRollStart) * y5 + imuShiftFromStartYCur;
  p->z = z5 + imuShiftFromStartZCur;
}

//Integral velocity and displacement
void AccumulateIMUShift()
{
  float roll = imuRoll[imuPointerLast];
  float pitch = imuPitch[imuPointerLast];
  float yaw = imuYaw[imuPointerLast];
  float accX = imuAccX[imuPointerLast];
  float accY = imuAccY[imuPointerLast];
  float accZ = imuAccZ[imuPointerLast];

  //Rotate the current acceleration value around the swapped ZXY fixed axis (original XYZ) by (roll, pitch, yaw) angles, and convert to the acceleration value in the world coordinate system (right hand rule)
  //Rotate around the z axis(roll)
  float x1 = cos(roll) * accX - sin(roll) * accY;
  float y1 = sin(roll) * accX + cos(roll) * accY;
  float z1 = accZ;
  //Rotate around the x axis(pitch)
  float x2 = x1;
  float y2 = cos(pitch) * y1 - sin(pitch) * z1;
  float z2 = sin(pitch) * y1 + cos(pitch) * z1;
  //Rotate around the y axis(yaw)
  accX = cos(yaw) * x2 + sin(yaw) * z2;
  accY = y2;
  accZ = -sin(yaw) * x2 + cos(yaw) * z2;

  //Previous imu point
  int imuPointerBack = (imuPointerLast + imuQueLength - 1) % imuQueLength;
  //The time elapsed from the last point to the current point, that is, the calculation of the imu measurement period
  double timeDiff = imuTime[imuPointerLast] - imuTime[imuPointerBack];
  //It is required that the frequency of imu is at least higher than lidar, so that the imu information is used, and the subsequent correction is also meaningful
  if (timeDiff < scanPeriod) {//（Implied start to move from stillness）
    //Find the displacement and velocity of each imu time point, between two points is regarded as a uniform acceleration linear motion
    imuShiftX[imuPointerLast] = imuShiftX[imuPointerBack] + imuVeloX[imuPointerBack] * timeDiff 
                              + accX * timeDiff * timeDiff / 2;
    imuShiftY[imuPointerLast] = imuShiftY[imuPointerBack] + imuVeloY[imuPointerBack] * timeDiff 
                              + accY * timeDiff * timeDiff / 2;
    imuShiftZ[imuPointerLast] = imuShiftZ[imuPointerBack] + imuVeloZ[imuPointerBack] * timeDiff 
                              + accZ * timeDiff * timeDiff / 2;

    imuVeloX[imuPointerLast] = imuVeloX[imuPointerBack] + accX * timeDiff;
    imuVeloY[imuPointerLast] = imuVeloY[imuPointerBack] + accY * timeDiff;
    imuVeloZ[imuPointerLast] = imuVeloZ[imuPointerBack] + accZ * timeDiff;
  }
}

//To receive point cloud data, the velodyne laser coordinate system is installed as a right-handed coordinate system with x-axis forward, y-axis left, and z-axis upward
void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg) // line 245 ~ 735
{
  if (!systemInited) {//Discard the first 20 point cloud data
    systemInitCount++;
    if (systemInitCount >= systemDelay) {
      systemInited = true;
    }
    return;
  }

  //Record the start and end index of the curvature point of each scan
  std::vector<int> scanStartInd(N_SCANS, 0); // vector <data type> 변수이름 // vector<int> v(5, 2); 2로 초기화된 5개의 원소를 가지는 vector v를 생성합니다
  std::vector<int> scanEndInd(N_SCANS, 0);
  
  //Current point cloud time
  double timeScanCur = laserCloudMsg->header.stamp.toSec(); // lasserCloudMsg는 laserCloudHandler function의 argu로 정의됨.
  pcl::PointCloud<pcl::PointXYZ> laserCloudIn; // header은 대체 뭐여? 뭔디 stamp랑 toSec이 있음? 이것들은 또 머임?
  //The message is converted into pcl data storage
  pcl::fromROSMsg(*laserCloudMsg, laserCloudIn);
  std::vector<int> indices; // indices vector 변수 생성
  //Remove empty spots
  pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn, indices);
  //Number of point cloud points
  int cloudSize = laserCloudIn.points.size();
  //The rotation angle of the starting point of lidar scan, atan2 range [-pi,+pi], the negative sign when calculating the rotation angle is because velodyne rotates clockwise
  float startOri = -atan2(laserCloudIn.points[0].y, laserCloudIn.points[0].x);
  //The rotation angle of the end point of lidar scan, add 2*pi to make the point cloud rotation period 2*pi
  float endOri = -atan2(laserCloudIn.points[cloudSize - 1].y,
                        laserCloudIn.points[cloudSize - 1].x) + 2 * M_PI;

  //The difference between the end azimuth and the start azimuth is controlled within the range of (PI,3*PI), allowing lidar to be not a circle scan
  //Within this range under normal circumstances: pi <endOri-startOri <3*pi, if abnormalities are corrected
  if (endOri - startOri > 3 * M_PI) {
    endOri -= 2 * M_PI;
  } else if (endOri - startOri < M_PI) {
    endOri += 2 * M_PI;
  }
  //Is the lidar scan line rotated more than halfway? 여기서 halfway의 기준이 머임?
  bool halfPassed = false;
  int count = cloudSize;
  PointType point;
  std::vector<pcl::PointCloud<PointType> > laserCloudScans(N_SCANS);
  for (int i = 0; i < cloudSize; i++) { // for문이 여기부터 410줄 까지 감
    //Coordinate axis exchange, the coordinate system of velodyne lidar is also converted to the right-handed coordinate system with the z-axis forward and the x-axis to the left
    point.x = laserCloudIn.points[i].y;
    point.y = laserCloudIn.points[i].z;
    point.z = laserCloudIn.points[i].x;

    //Calculate the elevation angle of the point (according to the vertical angle calculation formula of the lidar document), arrange the laser line numbers according to the elevation angle, and the interval between every two scans of velodyne is 2 degrees
    float angle = atan(point.y / sqrt(point.x * point.x + point.z * point.z)) * 180 / M_PI; // 이게 radian to degree 공식임
    int scanID;
    //Elevation angle is rounded (plus or minus 0.5 truncation effect is equal to rounding)
    int roundedAngle = int(angle + (angle<0.0?-0.5:+0.5)); 
    if (roundedAngle > 0){
      scanID = roundedAngle; // 어떻게 Angle이 scanID가 되지?
    }
    else {
      scanID = roundedAngle + (N_SCANS - 1);
    }
    //Filter points, only select points within the range of [-15 degrees, +15 degrees], scanID belongs to [0,15]
    if (scanID > (N_SCANS - 1) || scanID < 0 ){ //이게 지금 원래 총 각도 30도 안에 point가 들어와야 하는데, error 등으로 30도 범위를 넘는 것들을 제하는 과정인듯.
      count--; // velodyne lidar vertical angle이 +-15도 해서 총 30도네
      continue;
    }

    //The rotation angle of the point
    float ori = -atan2(point.x, point.z); // point y가 위 방향이므로, 지금 여기서의 rotation angle은 2차원에서의 angle을 의미
    if (!halfPassed) {//According to whether the scan line is rotated more than halfway, the difference is calculated from the start position or the end position to compensate
        //Make sure -pi/2 <ori-startOri <3*pi/2
      if (ori < startOri - M_PI / 2) {
        ori += 2 * M_PI;
      } else if (ori > startOri + M_PI * 3 / 2) {
        ori -= 2 * M_PI;
      }

      if (ori - startOri > M_PI) {
        halfPassed = true;
      }
    } else {
      ori += 2 * M_PI;

      //Ensure -3*pi/2 <ori-endOri <pi/2 // 지금 여기 부분이 Ori의 방향 조정하는 건지는 알겠는데 구체적 의미는 안 와닿네
      if (ori < endOri - M_PI * 3 / 2) {
        ori += 2 * M_PI;
      } else if (ori > endOri + M_PI / 2) {
        ori -= 2 * M_PI;
      } 
    }

    //-0.5 <relTime <1.5 (the ratio of the angle of the point rotation to the angle of the entire cycle, that is, the relative time of the point in the point cloud)
    float relTime = (ori - startOri) / (endOri - startOri); //등속 운동이므로 상대적 시간 계산 되는 거임.
    //Point intensity = line number + point relative time (that is, an integer + a decimal, the integer part is the line number, and the decimal part is the relative time of the point), constant speed scanning: the relative scanning start position is calculated according to the current scanning angle and scanning period time
    point.intensity = scanID + scanPeriod * relTime; // 애초에 pointType이 XYZI가 아니라 XYZ로 위에서 정의 됐는데, 어케 intensity use?

    //Point time = point cloud time + cycle time
    if (imuPointerLast >= 0) {//If IMU data is received, use IMU to correct point cloud distortion
      float pointTime = relTime * scanPeriod;//Cycle time of calculation points
      //Find if the time stamp of the cloud is smaller than the IMU location of the IMU time stamp: imuPointerFront
      while (imuPointerFront != imuPointerLast) {
        if (timeScanCur + pointTime < imuTime[imuPointerFront]) {
          break;
        }
        imuPointerFront = (imuPointerFront + 1) % imuQueLength;
      }

      if (timeScanCur + pointTime > imuTime[imuPointerFront]) {//Not found, at this time imuPointerFront==imtPointerLast, you can only use the speed, displacement, and Euler angle of the latest IMU currently received as the speed, displacement, and Euler angle of the current point.
        imuRollCur = imuRoll[imuPointerFront];
        imuPitchCur = imuPitch[imuPointerFront];
        imuYawCur = imuYaw[imuPointerFront];

        imuVeloXCur = imuVeloX[imuPointerFront];
        imuVeloYCur = imuVeloY[imuPointerFront];
        imuVeloZCur = imuVeloZ[imuPointerFront];

        imuShiftXCur = imuShiftX[imuPointerFront];
        imuShiftYCur = imuShiftY[imuPointerFront];
        imuShiftZCur = imuShiftZ[imuPointerFront];
      } else {//Find the IMU position where the point cloud timestamp is less than the IMU timestamp, then the point must be between imuPointerBack and imuPointerFront. Based on this linear interpolation, calculate the point cloud point velocity, displacement and Euler angle
        int imuPointerBack = (imuPointerFront + imuQueLength - 1) % imuQueLength;
        //Calculate the weight distribution ratio based on time distance, that is, linear interpolation
        float ratioFront = (timeScanCur + pointTime - imuTime[imuPointerBack]) 
                         / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
        float ratioBack = (imuTime[imuPointerFront] - timeScanCur - pointTime) 
                        / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);

        imuRollCur = imuRoll[imuPointerFront] * ratioFront + imuRoll[imuPointerBack] * ratioBack;
        imuPitchCur = imuPitch[imuPointerFront] * ratioFront + imuPitch[imuPointerBack] * ratioBack;
        if (imuYaw[imuPointerFront] - imuYaw[imuPointerBack] > M_PI) {
          imuYawCur = imuYaw[imuPointerFront] * ratioFront + (imuYaw[imuPointerBack] + 2 * M_PI) * ratioBack;
        } else if (imuYaw[imuPointerFront] - imuYaw[imuPointerBack] < -M_PI) {
          imuYawCur = imuYaw[imuPointerFront] * ratioFront + (imuYaw[imuPointerBack] - 2 * M_PI) * ratioBack;
        } else {
          imuYawCur = imuYaw[imuPointerFront] * ratioFront + imuYaw[imuPointerBack] * ratioBack;
        }

        //Nature:imuVeloXCur = imuVeloX[imuPointerback] + (imuVelX[imuPointerFront]-imuVelX[imuPoniterBack])*ratioFront
        imuVeloXCur = imuVeloX[imuPointerFront] * ratioFront + imuVeloX[imuPointerBack] * ratioBack;
        imuVeloYCur = imuVeloY[imuPointerFront] * ratioFront + imuVeloY[imuPointerBack] * ratioBack;
        imuVeloZCur = imuVeloZ[imuPointerFront] * ratioFront + imuVeloZ[imuPointerBack] * ratioBack;

        imuShiftXCur = imuShiftX[imuPointerFront] * ratioFront + imuShiftX[imuPointerBack] * ratioBack;
        imuShiftYCur = imuShiftY[imuPointerFront] * ratioFront + imuShiftY[imuPointerBack] * ratioBack;
        imuShiftZCur = imuShiftZ[imuPointerFront] * ratioFront + imuShiftZ[imuPointerBack] * ratioBack;
      }

      if (i == 0) {//If it is the first point, remember the speed, displacement, Euler angle of the starting position of the point cloud
        imuRollStart = imuRollCur;
        imuPitchStart = imuPitchCur;
        imuYawStart = imuYawCur;

        imuVeloXStart = imuVeloXCur;
        imuVeloYStart = imuVeloYCur;
        imuVeloZStart = imuVeloZCur;

        imuShiftXStart = imuShiftXCur;
        imuShiftYStart = imuShiftYCur;
        imuShiftZStart = imuShiftZCur;
      } else {//After calculation, the displacement velocity distortion of each point relative to the first point due to the acceleration and deceleration non-uniform motion is calculated, and the position information of each point in the point cloud is recompensated and corrected
        ShiftToStartIMU(pointTime);
        VeloToStartIMU();
        TransformToStartIMU(&point);
      }
    }
    laserCloudScans[scanID].push_back(point);//Put each point of compensation correction into the container of the corresponding line number
  }

  //Get the number of points in the effective range
  cloudSize = count;

  pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());
  for (int i = 0; i < N_SCANS; i++) {//Put all the points in a container according to the line number from small to large
    *laserCloud += laserCloudScans[i];
  }
  int scanCount = -1;
  for (int i = 5; i < cloudSize - 5; i++) {//Use the five points before and after each point to calculate the curvature, so the first five and last five points are skipped
    float diffX = laserCloud->points[i - 5].x + laserCloud->points[i - 4].x 
                + laserCloud->points[i - 3].x + laserCloud->points[i - 2].x 
                + laserCloud->points[i - 1].x - 10 * laserCloud->points[i].x 
                + laserCloud->points[i + 1].x + laserCloud->points[i + 2].x
                + laserCloud->points[i + 3].x + laserCloud->points[i + 4].x
                + laserCloud->points[i + 5].x;
    float diffY = laserCloud->points[i - 5].y + laserCloud->points[i - 4].y 
                + laserCloud->points[i - 3].y + laserCloud->points[i - 2].y 
                + laserCloud->points[i - 1].y - 10 * laserCloud->points[i].y 
                + laserCloud->points[i + 1].y + laserCloud->points[i + 2].y
                + laserCloud->points[i + 3].y + laserCloud->points[i + 4].y
                + laserCloud->points[i + 5].y;
    float diffZ = laserCloud->points[i - 5].z + laserCloud->points[i - 4].z 
                + laserCloud->points[i - 3].z + laserCloud->points[i - 2].z 
                + laserCloud->points[i - 1].z - 10 * laserCloud->points[i].z 
                + laserCloud->points[i + 1].z + laserCloud->points[i + 2].z
                + laserCloud->points[i + 3].z + laserCloud->points[i + 4].z
                + laserCloud->points[i + 5].z;
    //Curvature calculation
    cloudCurvature[i] = diffX * diffX + diffY * diffY + diffZ * diffZ;
    //Record the index of the curvature point
    cloudSortInd[i] = i;
    //Initially, the points are not filtered
    cloudNeighborPicked[i] = 0;
    //Initialize to less flat point
    cloudLabel[i] = 0;

    //For each scan, only the first matching point will come in, because the points of each scan are stored together
    if (int(laserCloud->points[i].intensity) != scanCount) {
      scanCount = int(laserCloud->points[i].intensity);//Control each scan to enter only the first point

      //The curvature is only calculated by the same scan, and the curvature calculated across scans is illegal. Exclude, that is, exclude the five points before and after each scan
      if (scanCount > 0 && scanCount < N_SCANS) {
        scanStartInd[scanCount] = i + 5;
        scanEndInd[scanCount - 1] = i - 5;
      }
    }
  }
  //The effective point sequence of the first scan curvature point starts from the 5th, and the last laser line ends the point sequence size-5
  scanStartInd[0] = 5;
  scanEndInd.back() = cloudSize - 5; // .back은 무슨 의미?

  //Select points and exclude points that are easily blocked by the slope and outliers. Some points are easily blocked by the slope, and the outliers may appear accidental. These situations may cause the two scans to be unable to be seen at the same time.
  for (int i = 5; i < cloudSize - 6; i++) {//The difference with the next point, so subtract 6
    float diffX = laserCloud->points[i + 1].x - laserCloud->points[i].x;
    float diffY = laserCloud->points[i + 1].y - laserCloud->points[i].y;
    float diffZ = laserCloud->points[i + 1].z - laserCloud->points[i].z;
    //Calculate the sum of squared distances between the effective curvature point and the next point
    float diff = diffX * diffX + diffY * diffY + diffZ * diffZ;

    if (diff > 0.1) {//Prerequisite: The distance between two points must be greater than 0.1
        //Point depth
      float depth1 = sqrt(laserCloud->points[i].x * laserCloud->points[i].x + 
                     laserCloud->points[i].y * laserCloud->points[i].y +
                     laserCloud->points[i].z * laserCloud->points[i].z);

      //The depth of the latter point
      float depth2 = sqrt(laserCloud->points[i + 1].x * laserCloud->points[i + 1].x + 
                     laserCloud->points[i + 1].y * laserCloud->points[i + 1].y +
                     laserCloud->points[i + 1].z * laserCloud->points[i + 1].z);

      //According to the ratio of the depth of the two points, draw the deeper point back and calculate the distance
      if (depth1 > depth2) {
        diffX = laserCloud->points[i + 1].x - laserCloud->points[i].x * depth2 / depth1;
        diffY = laserCloud->points[i + 1].y - laserCloud->points[i].y * depth2 / depth1;
        diffZ = laserCloud->points[i + 1].z - laserCloud->points[i].z * depth2 / depth1;

        //The side length ratio is also the radian value. If it is less than 0.1, it means that the included angle is relatively small, the slope is relatively steep, the point depth changes more drastically, and the point is on a slope that is approximately parallel to the laser beam.
        if (sqrt(diffX * diffX + diffY * diffY + diffZ * diffZ) / depth2 < 0.1) {//排除容易被斜面挡住的点
            //This point and the previous five points (approximately on the slope) are all set to be filtered
          cloudNeighborPicked[i - 5] = 1;
          cloudNeighborPicked[i - 4] = 1;
          cloudNeighborPicked[i - 3] = 1;
          cloudNeighborPicked[i - 2] = 1;
          cloudNeighborPicked[i - 1] = 1;
          cloudNeighborPicked[i] = 1;
        }
      } else {
        diffX = laserCloud->points[i + 1].x * depth1 / depth2 - laserCloud->points[i].x;
        diffY = laserCloud->points[i + 1].y * depth1 / depth2 - laserCloud->points[i].y;
        diffZ = laserCloud->points[i + 1].z * depth1 / depth2 - laserCloud->points[i].z;

        if (sqrt(diffX * diffX + diffY * diffY + diffZ * diffZ) / depth1 < 0.1) {
          cloudNeighborPicked[i + 1] = 1;
          cloudNeighborPicked[i + 2] = 1;
          cloudNeighborPicked[i + 3] = 1;
          cloudNeighborPicked[i + 4] = 1;
          cloudNeighborPicked[i + 5] = 1;
          cloudNeighborPicked[i + 6] = 1;
        }
      }
    }

    float diffX2 = laserCloud->points[i].x - laserCloud->points[i - 1].x;
    float diffY2 = laserCloud->points[i].y - laserCloud->points[i - 1].y;
    float diffZ2 = laserCloud->points[i].z - laserCloud->points[i - 1].z;
    //The sum of squares of the distance from the previous point
    float diff2 = diffX2 * diffX2 + diffY2 * diffY2 + diffZ2 * diffZ2;

    //Sum of squares of point depth
    float dis = laserCloud->points[i].x * laserCloud->points[i].x
              + laserCloud->points[i].y * laserCloud->points[i].y
              + laserCloud->points[i].z * laserCloud->points[i].z;

    //The sum of squares with the points before and after are greater than two ten thousandths of the sum of squares of depth. These points are regarded as outliers, including points on steep slopes, strong convexities and concave points, and certain points in open areas. They are set as filtered and discarded. use
    if (diff > 0.0002 * dis && diff2 > 0.0002 * dis) {
      cloudNeighborPicked[i] = 1;
    }
  }


  pcl::PointCloud<PointType> cornerPointsSharp;
  pcl::PointCloud<PointType> cornerPointsLessSharp;
  pcl::PointCloud<PointType> surfPointsFlat;
  pcl::PointCloud<PointType> surfPointsLessFlat;

  //Divide the points on each line into corresponding categories: edge points and plane points
  for (int i = 0; i < N_SCANS; i++) {
    pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<PointType>);
    //Divide the curvature points of each scan into 6 equal parts to ensure that there are points around are selected as feature points
    for (int j = 0; j < 6; j++) {
        //Hexadecimal starting point：sp = scanStartInd + (scanEndInd - scanStartInd)*j/6
      int sp = (scanStartInd[i] * (6 - j)  + scanEndInd[i] * j) / 6;
      //Hexadecimal end point：ep = scanStartInd - 1 + (scanEndInd - scanStartInd)*(j+1)/6
      int ep = (scanStartInd[i] * (5 - j)  + scanEndInd[i] * (j + 1)) / 6 - 1;

      //Bubbles sorted by curvature from small to large // bubble? 번역이 잘못 된듯
      for (int k = sp + 1; k <= ep; k++) {
        for (int l = k; l >= sp + 1; l--) {
            //If the back curvature point is greater than the front, then swap
          if (cloudCurvature[cloudSortInd[l]] < cloudCurvature[cloudSortInd[l - 1]]) {
            int temp = cloudSortInd[l - 1];
            cloudSortInd[l - 1] = cloudSortInd[l];
            cloudSortInd[l] = temp;
          }
        }
      }

      //Pick points with large and relatively large curvatures for each segment
      int largestPickedNum = 0;
      for (int k = ep; k >= sp; k--) {
        int ind = cloudSortInd[k];  //Point order of the point of maximum curvature
// curvature의 뜻을 확실히 알아야 할듯
        //If the curvature is large, the curvature is indeed relatively large, and it is not filtered out
        if (cloudNeighborPicked[ind] == 0 &&
            cloudCurvature[ind] > 0.1) {
        
          largestPickedNum++;
          if (largestPickedNum <= 2) {//Pick the first 2 points with the largest curvature into the sharp point set
            cloudLabel[ind] = 2;//2 means the point has a large curvature
            cornerPointsSharp.push_back(laserCloud->points[ind]);
            cornerPointsLessSharp.push_back(laserCloud->points[ind]);
          } else if (largestPickedNum <= 20) {//Pick the first 20 points with the largest curvature into the less sharp point set
            cloudLabel[ind] = 1;//1 means that the curvature of the point is relatively sharp
            cornerPointsLessSharp.push_back(laserCloud->points[ind]);
          } else {
            break;
          }

          cloudNeighborPicked[ind] = 1;//Filter flag set

          //Filter out 5 consecutive points with relatively close distance before and after the point with relatively large curvature to prevent the feature points from clustering, so that the feature points are distributed as evenly as possible in each direction
          for (int l = 1; l <= 5; l++) {
            float diffX = laserCloud->points[ind + l].x 
                        - laserCloud->points[ind + l - 1].x;
            float diffY = laserCloud->points[ind + l].y 
                        - laserCloud->points[ind + l - 1].y;
            float diffZ = laserCloud->points[ind + l].z 
                        - laserCloud->points[ind + l - 1].z;
            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
              break;
            }

            cloudNeighborPicked[ind + l] = 1;
          }
          for (int l = -1; l >= -5; l--) {
            float diffX = laserCloud->points[ind + l].x 
                        - laserCloud->points[ind + l + 1].x;
            float diffY = laserCloud->points[ind + l].y 
                        - laserCloud->points[ind + l + 1].y;
            float diffZ = laserCloud->points[ind + l].z 
                        - laserCloud->points[ind + l + 1].z;
            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
              break;
            }

            cloudNeighborPicked[ind + l] = 1;
          }
        }
      }

      //Pick the point where the curvature of each segment is very small
      int smallestPickedNum = 0;
      for (int k = sp; k <= ep; k++) {
        int ind = cloudSortInd[k];

        //If the curvature is indeed small and has not been filtered out
        if (cloudNeighborPicked[ind] == 0 &&
            cloudCurvature[ind] < 0.1) {

          cloudLabel[ind] = -1;//-1 represents a point with a small curvature
          surfPointsFlat.push_back(laserCloud->points[ind]);

          smallestPickedNum++;
          if (smallestPickedNum >= 4) {//Only choose the smallest four, the remaining Label==0, all have relatively small curvature
            break;
          }

          cloudNeighborPicked[ind] = 1;
          for (int l = 1; l <= 5; l++) {//Also prevent feature points from clustering
            float diffX = laserCloud->points[ind + l].x 
                        - laserCloud->points[ind + l - 1].x;
            float diffY = laserCloud->points[ind + l].y 
                        - laserCloud->points[ind + l - 1].y;
            float diffZ = laserCloud->points[ind + l].z 
                        - laserCloud->points[ind + l - 1].z;
            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
              break;
            }

            cloudNeighborPicked[ind + l] = 1;
          }
          for (int l = -1; l >= -5; l--) {
            float diffX = laserCloud->points[ind + l].x 
                        - laserCloud->points[ind + l + 1].x;
            float diffY = laserCloud->points[ind + l].y 
                        - laserCloud->points[ind + l + 1].y;
            float diffZ = laserCloud->points[ind + l].z 
                        - laserCloud->points[ind + l + 1].z;
            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05) {
              break;
            }

            cloudNeighborPicked[ind + l] = 1;
          }
        }
      }

      //All the remaining points (including the previously excluded points) are classified into the less flat category in the plane point
      for (int k = sp; k <= ep; k++) {
        if (cloudLabel[k] <= 0) {
          surfPointsLessFlatScan->push_back(laserCloud->points[k]);
        }
      }
    }

    //Since there are the most less flat points, voxel grid filtering is performed on the less flat points of each segment
    pcl::PointCloud<PointType> surfPointsLessFlatScanDS; // 여기서 voxel grid filtering이 등판!
    pcl::VoxelGrid<PointType> downSizeFilter;
    downSizeFilter.setInputCloud(surfPointsLessFlatScan);
    downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
    downSizeFilter.filter(surfPointsLessFlatScanDS);

    //less flat point summary
    surfPointsLessFlat += surfPointsLessFlatScanDS;
  }

  //publish eliminates all points after non-uniform motion distortion
  sensor_msgs::PointCloud2 laserCloudOutMsg;
  pcl::toROSMsg(*laserCloud, laserCloudOutMsg);
  laserCloudOutMsg.header.stamp = laserCloudMsg->header.stamp;
  laserCloudOutMsg.header.frame_id = "/camera";
  pubLaserCloud.publish(laserCloudOutMsg);

  //publish eliminates plane points and edge points after non-uniform motion distortion
  sensor_msgs::PointCloud2 cornerPointsSharpMsg;
  pcl::toROSMsg(cornerPointsSharp, cornerPointsSharpMsg);
  cornerPointsSharpMsg.header.stamp = laserCloudMsg->header.stamp;
  cornerPointsSharpMsg.header.frame_id = "/camera";
  pubCornerPointsSharp.publish(cornerPointsSharpMsg);

  sensor_msgs::PointCloud2 cornerPointsLessSharpMsg;
  pcl::toROSMsg(cornerPointsLessSharp, cornerPointsLessSharpMsg);
  cornerPointsLessSharpMsg.header.stamp = laserCloudMsg->header.stamp;
  cornerPointsLessSharpMsg.header.frame_id = "/camera";
  pubCornerPointsLessSharp.publish(cornerPointsLessSharpMsg);

  sensor_msgs::PointCloud2 surfPointsFlat2;
  pcl::toROSMsg(surfPointsFlat, surfPointsFlat2);
  surfPointsFlat2.header.stamp = laserCloudMsg->header.stamp;
  surfPointsFlat2.header.frame_id = "/camera";
  pubSurfPointsFlat.publish(surfPointsFlat2);

  sensor_msgs::PointCloud2 surfPointsLessFlat2;
  pcl::toROSMsg(surfPointsLessFlat, surfPointsLessFlat2);
  surfPointsLessFlat2.header.stamp = laserCloudMsg->header.stamp;
  surfPointsLessFlat2.header.frame_id = "/camera";
  pubSurfPointsLessFlat.publish(surfPointsLessFlat2);

  //The publish IMU message, because the loop is at the end, Cur represents the last point, that is, the Euler angle of the last point, the distortion displacement and the speed of a point cloud cycle increase
  pcl::PointCloud<pcl::PointXYZ> imuTrans(4, 1);
  //Starting point Euler angle
  imuTrans.points[0].x = imuPitchStart;
  imuTrans.points[0].y = imuYawStart;
  imuTrans.points[0].z = imuRollStart;

  //Euler angle at the last point
  imuTrans.points[1].x = imuPitchCur;
  imuTrans.points[1].y = imuYawCur;
  imuTrans.points[1].z = imuRollCur;

  //Distortion displacement and velocity of the last point relative to the first point
  imuTrans.points[2].x = imuShiftFromStartXCur;
  imuTrans.points[2].y = imuShiftFromStartYCur;
  imuTrans.points[2].z = imuShiftFromStartZCur;

  imuTrans.points[3].x = imuVeloFromStartXCur;
  imuTrans.points[3].y = imuVeloFromStartYCur;
  imuTrans.points[3].z = imuVeloFromStartZCur;

  sensor_msgs::PointCloud2 imuTransMsg;
  pcl::toROSMsg(imuTrans, imuTransMsg);
  imuTransMsg.header.stamp = laserCloudMsg->header.stamp;
  imuTransMsg.header.frame_id = "/camera";
  pubImuTrans.publish(imuTransMsg);
}

//Receive imu message, the imu coordinate system is a right-handed coordinate system with x axis forward, y axis right, and z axis up
void imuHandler(const sensor_msgs::Imu::ConstPtr& imuIn)
{
  double roll, pitch, yaw;
  tf::Quaternion orientation;
  //convert Quaternion msg to Quaternion // quaternion = 사원수
  tf::quaternionMsgToTF(imuIn->orientation, orientation);
  //This will get the roll pitch and yaw from the matrix about fixed axes X, Y, Z respectively. That's R = Rz(yaw)*Ry(pitch)*Rx(roll).
  //Here roll pitch yaw is in the global frame
  tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

  //Subtract the influence of gravity, calculate the actual acceleration value in the xyz direction, and exchange the coordinate axes, unified to the right-hand coordinate system of the z-axis forward and the x-axis left. After the exchange, RPY corresponds to fixed axes ZXY(RPY---ZXY ). Now R = Ry(yaw)*Rx(pitch)*Rz(roll).
  float accX = imuIn->linear_acceleration.y - sin(roll) * cos(pitch) * 9.81;
  float accY = imuIn->linear_acceleration.z - cos(roll) * cos(pitch) * 9.81;
  float accZ = imuIn->linear_acceleration.x + sin(pitch) * 9.81;

  //Circular shift effect, forming a circular array
  imuPointerLast = (imuPointerLast + 1) % imuQueLength;

  imuTime[imuPointerLast] = imuIn->header.stamp.toSec();
  imuRoll[imuPointerLast] = roll;
  imuPitch[imuPointerLast] = pitch;
  imuYaw[imuPointerLast] = yaw;
  imuAccX[imuPointerLast] = accX;
  imuAccY[imuPointerLast] = accY;
  imuAccZ[imuPointerLast] = accZ;

  AccumulateIMUShift();
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "scanRegistration");
  ros::NodeHandle nh; // ros 책에서 NodeHandle이 뭔지 정확히 알아봐야 겠군.

  ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2> 
                                  ("/velodyne_points", 2, laserCloudHandler);

  ros::Subscriber subImu = nh.subscribe<sensor_msgs::Imu> ("/imu/data", 50, imuHandler);

  pubLaserCloud = nh.advertise<sensor_msgs::PointCloud2>
                                 ("/velodyne_cloud_2", 2);

  pubCornerPointsSharp = nh.advertise<sensor_msgs::PointCloud2>
                                        ("/laser_cloud_sharp", 2);

  pubCornerPointsLessSharp = nh.advertise<sensor_msgs::PointCloud2>
                                            ("/laser_cloud_less_sharp", 2);

  pubSurfPointsFlat = nh.advertise<sensor_msgs::PointCloud2>
                                       ("/laser_cloud_flat", 2);

  pubSurfPointsLessFlat = nh.advertise<sensor_msgs::PointCloud2>
                                           ("/laser_cloud_less_flat", 2);

  pubImuTrans = nh.advertise<sensor_msgs::PointCloud2> ("/imu_trans", 5);

  ros::spin();

  return 0;
}

