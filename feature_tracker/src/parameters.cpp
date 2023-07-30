#include "parameters.h"

std::string IMAGE_TOPIC;
std::string IMU_TOPIC;
std::vector<std::string> CAM_NAMES;
std::string FISHEYE_MASK;
int MAX_CNT;
int MIN_DIST;
int WINDOW_SIZE;
int FREQ;
double F_THRESHOLD;
int SHOW_TRACK;
int STEREO_TRACK;
int EQUALIZE;
int ROW;
int COL;
int FOCAL_LENGTH;
int FISHEYE;
bool PUB_THIS_FRAME;

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
    std::string VINS_FOLDER_PATH = readParam<std::string>(n, "vins_folder"); //vins_folder 경로 저장

    fsSettings["image_topic"] >> IMAGE_TOPIC; //이미지의 토픽 저장 (cam0/image_raw)
    fsSettings["imu_topic"] >> IMU_TOPIC; // imu의 토픽 저장 (imu0/)
    MAX_CNT = fsSettings["max_cnt"]; //tracking시 추출되는 최대 feature 수 (150) 
    MIN_DIST = fsSettings["min_dist"]; //2개의 feature 최소 거리 (30) 
    ROW = fsSettings["image_height"]; //이미지의 row
    COL = fsSettings["image_width"]; //이미지의 col
    FREQ = fsSettings["freq"]; //tracking 빈도 수 (10HZ)
    F_THRESHOLD = fsSettings["F_threshold"]; //RANSAC threshold 값 (1 pixel)
    SHOW_TRACK = fsSettings["show_track"]; // tracking 시 이미지를 확인할 것인지에 대한 플레그
    EQUALIZE = fsSettings["equalize"]; //이미지가 너무 밝거나 어두울 경우를 위한 플레그
    FISHEYE = fsSettings["fisheye"]; //피시아이 카메라 사용 시
    if (FISHEYE == 1)
        FISHEYE_MASK = VINS_FOLDER_PATH + "config/fisheye_mask.jpg";
    CAM_NAMES.push_back(config_file); //config_file 이름 배열에 저장

    WINDOW_SIZE = 20; //window 크기 지정
    STEREO_TRACK = false; //스테레오인 경우 (mono이므로 필요없는 변수)
    FOCAL_LENGTH = 460;
    PUB_THIS_FRAME = false;

    if (FREQ == 0)
        FREQ = 100;

    fsSettings.release();


}
