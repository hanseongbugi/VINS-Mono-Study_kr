#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Bool.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>

#include "feature_tracker.h"

#define SHOW_UNDISTORTION 0

vector<uchar> r_status;
vector<float> r_err;
queue<sensor_msgs::ImageConstPtr> img_buf;

ros::Publisher pub_img,pub_match;
ros::Publisher pub_restart;

FeatureTracker trackerData[NUM_OF_CAM];
double first_image_time;
int pub_count = 1;
bool first_image_flag = true;
double last_image_time = 0;
bool init_pub = 0;

void img_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    if(first_image_flag) // 첫 frame 플레그가 true인 경우
    { 
        first_image_flag = false; // 첫 frame이 들어왔기에 false로 설정
        first_image_time = img_msg->header.stamp.toSec(); // 첫 frame의 time stamp를 저장
        last_image_time = img_msg->header.stamp.toSec(); // 이전 frame의 time stamp를 초기화
        return; // 함수 종료
    }
    // detect unstable camera stream
    // 현재 frame의 time stamp가 이전 frame의 time stamp보다 1초 이후에 들어왔거나 현재 frame이 이전 프레임보다 이전에 들어온 경우
    if (img_msg->header.stamp.toSec() - last_image_time > 1.0 || img_msg->header.stamp.toSec() < last_image_time)
    {
        ROS_WARN("image discontinue! reset the feature tracker!");
        first_image_flag = true;  // feature tracker 초기화를 위해 flag 초기화
        last_image_time = 0;
        pub_count = 1;
        std_msgs::Bool restart_flag;
        restart_flag.data = true;
        pub_restart.publish(restart_flag); // restart flag를 node에 전송 (estimator 등에 전송함)
        return;
    }
    last_image_time = img_msg->header.stamp.toSec(); // 이전 frame의 time stamp를 갱신
    // frequency control
    // node에 frame이 들어온 횟수 / (현재 frame의 time stmap - 첫 frame의 time stamp)의 올림 연산의 결과가 tracking 빈도 수 (10HZ) 이하인 경우
    if (round(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time)) <= FREQ) 
    {
        PUB_THIS_FRAME = true; // 현재 frame은 frequency 안에 있기에 true로 설정
        // reset the frequency control
        // node에 frame이 들어온 횟수 / (현재 frame의 time stmap - 첫 frame의 time stamp) - Freqency의 절대 값이 tracking 빈도 수 * 0.01 미만인 경우
        if (abs(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time) - FREQ) < 0.01 * FREQ)
        {
            first_image_time = img_msg->header.stamp.toSec(); // 첫 프레임의 time stamp를 현재 frame이 timestamp로 갱신
            pub_count = 0; // node에 frame이 들이온 횟수 초기화 
        }
    }
    else
        PUB_THIS_FRAME = false; // publish하지 않는 frame으로 설정

    cv_bridge::CvImageConstPtr ptr; //ROS에서 이미지 메시지와 OpenCV 이미지 간의 변환을 수행하는 데 사용되는 포인터를 생성
    if (img_msg->encoding == "8UC1") // 메시지의 인코딩 방식이 8UC1(8비트 unsigned char 형식의 흑백 이미지)인 경우
    {
        sensor_msgs::Image img;
        img.header = img_msg->header; // 메시지에 있는 정보를 OpenCv로 복사할 수 있는 형태로 변환
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8); // frame 복사
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8); // frame 복사

    cv::Mat show_img = ptr->image; // Mat 객체에 frame 복사
    TicToc t_r;
    for (int i = 0; i < NUM_OF_CAM; i++) // 카메라 수만큼 반복
    {
        ROS_DEBUG("processing camera %d", i);
        if (i != 1 || !STEREO_TRACK) // mono인 경우
            trackerData[i].readImage(ptr->image.rowRange(ROW * i, ROW * (i + 1)), img_msg->header.stamp.toSec()); // FeatureTracker 객체의 형태로 배열에 저장
        else // 이경우는 사용하지 않음
        {
            if (EQUALIZE)
            {
                cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
                clahe->apply(ptr->image.rowRange(ROW * i, ROW * (i + 1)), trackerData[i].cur_img);
            }
            else
                trackerData[i].cur_img = ptr->image.rowRange(ROW * i, ROW * (i + 1));
        }

#if SHOW_UNDISTORTION 
        trackerData[i].showUndistortion("undistrotion_" + std::to_string(i)); // SHOW_UNDISTORITION을 0으로 정의하였기에 호출되지 않는 부분, 하지만 1로 설정하고 빌드하면 왜곡되지 않은 이미지를 볼 수 있음
#endif
    }

    for (unsigned int i = 0;; i++) // 무한 loop
    {
        bool completed = false; // loop stop Flag
        for (int j = 0; j < NUM_OF_CAM; j++) // 카메라 수만큼 반복
            if (j != 1 || !STEREO_TRACK) // mono인 경우
                completed |= trackerData[j].updateID(i); // feature의 index를 증가시킴
        if (!completed)
            break; // feature index 증가 시 루프 종료
    }

   if (PUB_THIS_FRAME) // 현재 frame이 publish해야하는 경우
   {
        pub_count++; // publish 횟수 증가
        sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud);
        sensor_msgs::ChannelFloat32 id_of_point;
        sensor_msgs::ChannelFloat32 u_of_point;
        sensor_msgs::ChannelFloat32 v_of_point;
        sensor_msgs::ChannelFloat32 velocity_x_of_point;
        sensor_msgs::ChannelFloat32 velocity_y_of_point;

        feature_points->header = img_msg->header; // feature_points의 header를 frame의 header로 저장
        feature_points->header.frame_id = "world";

        vector<set<int>> hash_ids(NUM_OF_CAM);
        for (int i = 0; i < NUM_OF_CAM; i++) // 카메라 개수 만큼 loop
        {
            auto &un_pts = trackerData[i].cur_un_pts; // 현재 undistorted feature point 배열
            auto &cur_pts = trackerData[i].cur_pts; // 현재 feature point 배열
            auto &ids = trackerData[i].ids; // 현재 feature의 index
            auto &pts_velocity = trackerData[i].pts_velocity; // 현재 frame의 속도
            for (unsigned int j = 0; j < ids.size(); j++)
            {
                if (trackerData[i].track_cnt[j] > 1) // tracking count가 1보다 크다면
                {
                    int p_id = ids[j]; // feature index 볶사
                    hash_ids[i].insert(p_id); // index를 hash에 삽입
                    geometry_msgs::Point32 p;
                    p.x = un_pts[j].x; // feature point를 저장
                    p.y = un_pts[j].y;
                    p.z = 1;

                    feature_points->points.push_back(p); // feature point를 배열에 삽입
                    id_of_point.values.push_back(p_id * NUM_OF_CAM + i); // feature index에 카메라 수를 곱하여 배열에 삽입
                    u_of_point.values.push_back(cur_pts[j].x); // feature의 x, y 값 삽입
                    v_of_point.values.push_back(cur_pts[j].y);
                    velocity_x_of_point.values.push_back(pts_velocity[j].x); // frame의 속도 삽입
                    velocity_y_of_point.values.push_back(pts_velocity[j].y);
                }
            }
        }
        feature_points->channels.push_back(id_of_point); // 생성한 배열을 publish하기 위해 객체에 저장
        feature_points->channels.push_back(u_of_point);
        feature_points->channels.push_back(v_of_point);
        feature_points->channels.push_back(velocity_x_of_point);
        feature_points->channels.push_back(velocity_y_of_point);
        ROS_DEBUG("publish %f, at %f", feature_points->header.stamp.toSec(), ros::Time::now().toSec());
        // skip the first image; since no optical speed on frist image
        if (!init_pub) // publish할 대상이 첫번 째 이미지인 경우
        {
            init_pub = 1; // publish가 초기화 되었다고 설정
        }
        else
            pub_img.publish(feature_points); // 생성한 객체를 publish

        if (SHOW_TRACK) // config파일에서 SHOW_TRACK을 1로 한 경우
        {
            ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8); // ROS 이미지 메시지를 opencv의 BGR 형식 이미지로 변환
            //cv::Mat stereo_img(ROW * NUM_OF_CAM, COL, CV_8UC3);
            cv::Mat stereo_img = ptr->image; // Mat 객체로 frame 저장

            for (int i = 0; i < NUM_OF_CAM; i++)
            {
                cv::Mat tmp_img = stereo_img.rowRange(i * ROW, (i + 1) * ROW); // tmp_img에 현재 카메라의 영역을 선택하여 저장
                cv::cvtColor(show_img, tmp_img, CV_GRAY2RGB); // show_img 이미지를 GRAY에서 BGR 형식으로 변환하여 tmp_img에 저장

                for (unsigned int j = 0; j < trackerData[i].cur_pts.size(); j++) // 현재 freature point 배열 크기만큼 loop
                {
                    double len = std::min(1.0, 1.0 * trackerData[i].track_cnt[j] / WINDOW_SIZE); // tracking count/ WINDOW_SIZE와 1중 작은 값을 len으로 설정
                    cv::circle(tmp_img, trackerData[i].cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2); // feature를 점으로 시각화
                    //draw speed line
                    /*
                    Vector2d tmp_cur_un_pts (trackerData[i].cur_un_pts[j].x, trackerData[i].cur_un_pts[j].y);
                    Vector2d tmp_pts_velocity (trackerData[i].pts_velocity[j].x, trackerData[i].pts_velocity[j].y);
                    Vector3d tmp_prev_un_pts;
                    tmp_prev_un_pts.head(2) = tmp_cur_un_pts - 0.10 * tmp_pts_velocity;
                    tmp_prev_un_pts.z() = 1;
                    Vector2d tmp_prev_uv;
                    trackerData[i].m_camera->spaceToPlane(tmp_prev_un_pts, tmp_prev_uv);
                    cv::line(tmp_img, trackerData[i].cur_pts[j], cv::Point2f(tmp_prev_uv.x(), tmp_prev_uv.y()), cv::Scalar(255 , 0, 0), 1 , 8, 0);
                    */
                    //char name[10];
                    //sprintf(name, "%d", trackerData[i].ids[j]);
                    //cv::putText(tmp_img, name, trackerData[i].cur_pts[j], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
                }
            }
            //cv::imshow("vis", stereo_img);
            //cv::waitKey(5);
            pub_match.publish(ptr->toImageMsg()); // 시각화한 이미지를 publish
        }
    }
    ROS_INFO("whole feature tracker processing costs: %f", t_r.toc());
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "feature_tracker"); // ROS node로 등록 (스레드를 만든다고 생각)
    ros::NodeHandle n("~"); //ROS 매개변수 서버의 로컬 namesapce로부터 매개 변수를 가져옴 (config파일 정보, vins_folder 정보)
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info); //ROS 로깅 레벨 설정 (프로그램의 실행 중에 발생하는 이벤트나 상태 정보를 기록, info는 일반적인 상태 정보)
    readParameters(n); //매개변수를 읽는다. (feature_tracker의 paramerter.cpp에 있는 readParameters 함수 호출)

    for (int i = 0; i < NUM_OF_CAM; i++) //단안이므로 loop는 1회만 반복함
        trackerData[i].readIntrinsicParameter(CAM_NAMES[i]); //config 파일을 읽어 FeatureTracker에 트레킹을 위한 파라미터를 저장함

    if(FISHEYE) //FishEye 카메라인 경우
    {
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            trackerData[i].fisheye_mask = cv::imread(FISHEYE_MASK, 0);
            if(!trackerData[i].fisheye_mask.data)
            {
                ROS_INFO("load mask fail");
                ROS_BREAK();
            }
            else
                ROS_INFO("load mask success");
        }
    }
    // /cam0/image_raw 라는 토픽을 구독하고, 메시지 큐 크기는 100으로 설정,새로운 메시지가 오면 img_callback함수를 호출(callback함수)
    ros::Subscriber sub_img = n.subscribe(IMAGE_TOPIC, 100, img_callback); 

    pub_img = n.advertise<sensor_msgs::PointCloud>("feature", 1000); //feature으로 publish할 토픽을 지정(받는 쪽은 /feature_tracker/feature), 메시지 큐 크기는 1000으로 지정. 이때 메시지 유형은 sensor_msgs::PointCloud이다.
    pub_match = n.advertise<sensor_msgs::Image>("feature_img",1000); //feature_img으로 publish할 토픽을 지정, 메시지 큐 크기는 1000으로 지정. 이때 메시지 유형은 sensor_msgs::Image이다.
    pub_restart = n.advertise<std_msgs::Bool>("restart",1000); //restart으로 publish할 토픽을 지정, 메시지 큐 크기는 1000으로 지정. 이때 메시지 유형은 std_msgs::Bool이다.
    /*
    if (SHOW_TRACK)
        cv::namedWindow("vis", cv::WINDOW_NORMAL);
    */
    ros::spin(); //프로그램이 종료될 때 까지 콜백 함수 처리를 기다림
    return 0;
}


// new points velocity is 0, pub or not?
// track cnt > 1 pub?