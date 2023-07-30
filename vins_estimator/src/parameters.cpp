#include "parameters.h"

double INIT_DEPTH;
double MIN_PARALLAX;
double ACC_N, ACC_W;
double GYR_N, GYR_W;

std::vector<Eigen::Matrix3d> RIC;
std::vector<Eigen::Vector3d> TIC;

Eigen::Vector3d G{0.0, 0.0, 9.8};

double BIAS_ACC_THRESHOLD;
double BIAS_GYR_THRESHOLD;
double SOLVER_TIME;
int NUM_ITERATIONS;
int ESTIMATE_EXTRINSIC;
int ESTIMATE_TD;
int ROLLING_SHUTTER;
std::string EX_CALIB_RESULT_PATH;
std::string VINS_RESULT_PATH;
std::string IMU_TOPIC;
double ROW, COL;
double TD, TR;

template <typename T>
T readParam(ros::NodeHandle &n, std::string name)
{
    T ans; 
    if (n.getParam(name, ans)) //Node Handle에서 name 값을 가진 파라미터를 가져와 ans에 저장
    {
        ROS_INFO_STREAM("Loaded " << name << ": " << ans); //로그에 ans를 출력 (주로 파일 경로)
    }
    else
    {
        ROS_ERROR_STREAM("Failed to load " << name);
        n.shutdown();
    }
    return ans;
}

void readParameters(ros::NodeHandle &n)
{
    std::string config_file; //config_file의 이름을 담을 변수
    config_file = readParam<std::string>(n, "config_file"); //config_file과 관련된 인자를 읽는다. (euroc_config.yaml 혹은 euroc_config_no_extrinsic.yaml)
    cv::FileStorage fsSettings(config_file, cv::FileStorage::READ); //파일을 읽는다.
    if(!fsSettings.isOpened()) //파일이 존재하지 않는다면
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }

    fsSettings["imu_topic"] >> IMU_TOPIC; //bag파일의 imu topic을 가져옴 (/imu0)

    SOLVER_TIME = fsSettings["max_solver_time"]; //최대 solver 반복 시간을 가져옴 (ms 단위)
    NUM_ITERATIONS = fsSettings["max_num_iterations"]; //최대 solver 반복 횟수를 가져옴 (8회)
    MIN_PARALLAX = fsSettings["keyframe_parallax"]; //keyFrame 선정을 위한 threshold 값 (10 pixel)
    MIN_PARALLAX = MIN_PARALLAX / FOCAL_LENGTH; // threshold 값에 460(초점 거리)을 나눈다. (헤더 파일에 값이 정의되어 있음)

    std::string OUTPUT_PATH;
    fsSettings["output_path"] >> OUTPUT_PATH; // 맵 정보가 저장될 디렉터리
    VINS_RESULT_PATH = OUTPUT_PATH + "/vins_result_no_loop.csv"; //vins 실행 후 저장될 파일
    std::cout << "result path " << VINS_RESULT_PATH << std::endl; //저장될 파일의 경로 출력

    // create folder if not exists
    FileSystemHelper::createDirectoryIfNotExists(OUTPUT_PATH.c_str()); //디렉터리 생성 (없다면)

    std::ofstream fout(VINS_RESULT_PATH, std::ios::out); //csv 파일을 생성
    fout.close();

    ACC_N = fsSettings["acc_n"]; //가속도 noise
    ACC_W = fsSettings["acc_w"]; //가속도 bias
    GYR_N = fsSettings["gyr_n"]; //자이로 noise
    GYR_W = fsSettings["gyr_w"]; //자이로 bias
    G.z() = fsSettings["g_norm"]; // 중력 가속도
    ROW = fsSettings["image_height"]; //이미지의 row를 가져옴
    COL = fsSettings["image_width"]; //이미지의 col을 가져옴
    ROS_INFO("ROW: %f COL: %f ", ROW, COL); //이미지의 row와 col을 ROS 로그 메시지를 통해 출력 ([ INFO] : ROW : COL : 형식으로 출력됨)

    ESTIMATE_EXTRINSIC = fsSettings["estimate_extrinsic"]; //IMU와 camera 사이의 T행렬을 추정할 것인지 값을 가져옴
    if (ESTIMATE_EXTRINSIC == 2) // 초기 값 없이 추정을 시작함 (변환 행렬을 구함)
    {
        ROS_WARN("have no prior about extrinsic param, calibrate extrinsic param");
        RIC.push_back(Eigen::Matrix3d::Identity()); //단위 행렬(3x3)을 생성해 배열에 삽입
        TIC.push_back(Eigen::Vector3d::Zero()); //영 행렬(3x1)을 생성하여 배열에 삽입
        EX_CALIB_RESULT_PATH = OUTPUT_PATH + "/extrinsic_parameter.csv"; //추정 결과가 저장될 파일 경로 지정

    }
    else 
    {
        if ( ESTIMATE_EXTRINSIC == 1) // yaml 파일에 있는 값을 토대로 추정을 함
        {
            ROS_WARN(" Optimize extrinsic param around initial guess!");
            EX_CALIB_RESULT_PATH = OUTPUT_PATH + "/extrinsic_parameter.csv"; //추정 결과가 저장될 파일 경로 지정
        }
        if (ESTIMATE_EXTRINSIC == 0) // yaml 파일에 있는 값을 그대로 사용
            ROS_WARN(" fix extrinsic param ");

        cv::Mat cv_R, cv_T;
        fsSettings["extrinsicRotation"] >> cv_R; //imu 회전 변환 행렬을 가져옴
        fsSettings["extrinsicTranslation"] >> cv_T; //imu 이동 변환 행렬을 가져옴
        Eigen::Matrix3d eigen_R;
        Eigen::Vector3d eigen_T;
        cv::cv2eigen(cv_R, eigen_R); //opencv의 Mat 객체를 eigen3의 Matrix3d 객체로 변환
        cv::cv2eigen(cv_T, eigen_T); //opencv의 Mat 객체를 eigen3의 Vector3d 객체로 변환
        Eigen::Quaterniond Q(eigen_R); //회전 행렬의 값을 쿼터니언으로 변환
        eigen_R = Q.normalized(); //쿼터니언 값을 normalized
        RIC.push_back(eigen_R); //배열에 삽입
        TIC.push_back(eigen_T);
        ROS_INFO_STREAM("Extrinsic_R : " << std::endl << RIC[0]); //로그를 통해 행렬을 출력
        ROS_INFO_STREAM("Extrinsic_T : " << std::endl << TIC[0].transpose());
        
    } 

    INIT_DEPTH = 5.0;
    BIAS_ACC_THRESHOLD = 0.1;
    BIAS_GYR_THRESHOLD = 0.1;

    TD = fsSettings["td"]; //imu 시간과 이미지 시간과의 차이 값
    ESTIMATE_TD = fsSettings["estimate_td"]; //이 값이 1이면 차이 값을 사용 (imu와 카메라의 시간 동기화가 되어 있지 않는 경우)
    if (ESTIMATE_TD)
        ROS_INFO_STREAM("Unsynchronized sensors, online estimate time offset, initial td: " << TD);
    else
        ROS_INFO_STREAM("Synchronized sensors, fix time offset: " << TD);

    ROLLING_SHUTTER = fsSettings["rolling_shutter"]; //롤링 셔터 카메라인 경우
    if (ROLLING_SHUTTER)
    {
        TR = fsSettings["rolling_shutter_tr"];
        ROS_INFO_STREAM("rolling shutter camera, read out time per line: " << TR);
    }
    else
    {
        TR = 0;
    }
    
    fsSettings.release();
}
