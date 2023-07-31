#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "estimator.h"
#include "parameters.h"
#include "utility/visualization.h"


Estimator estimator;

std::condition_variable con;
double current_time = -1;
queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
queue<sensor_msgs::PointCloudConstPtr> relo_buf;
int sum_of_wait = 0;

std::mutex m_buf;
std::mutex m_state;
std::mutex i_buf;
std::mutex m_estimator;

double latest_time;
Eigen::Vector3d tmp_P;
Eigen::Quaterniond tmp_Q;
Eigen::Vector3d tmp_V;
Eigen::Vector3d tmp_Ba;
Eigen::Vector3d tmp_Bg;
Eigen::Vector3d acc_0;
Eigen::Vector3d gyr_0;
bool init_feature = 0;
bool init_imu = 1;
double last_imu_t = 0;

void predict(const sensor_msgs::ImuConstPtr &imu_msg)
{
    double t = imu_msg->header.stamp.toSec();
    if (init_imu) //초기화 상태이면
    {
        latest_time = t; //마지막 시간을 현재 imu timestamp로 초기화
        init_imu = 0; //더이상 초기화 하지 않도록 플레그를 0으로 변경
        return; //함수 종료
    }
    double dt = t - latest_time; //timestamp간 간격을 구함
    latest_time = t; //마지막 시간을 현재 imu timestamp로 초기화

    double dx = imu_msg->linear_acceleration.x; //timestamp에 해당하는 가속도 센서 값을 가져온다.
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    Eigen::Vector3d linear_acceleration{dx, dy, dz}; //가져온 값들을 묶어 3x1 행렬로 만든다.

    double rx = imu_msg->angular_velocity.x; //timestamp에 해당하는 자이로 센서 값을 가져온다.
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Eigen::Vector3d angular_velocity{rx, ry, rz}; //가져온 값들을 묶어 3x1 행렬로 만든다.

    Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba) - estimator.g; // 이전에 측정한 가속도 값 acc_0에서 Bias를 보정한 값

    Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg; //  IMU에서 측정한 자이로 값 gyr_0과 각속도의 평균을 구하고 Bias만큼 보정
    tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt); // 자이로 값에서 imu 센서의 방향을 나타내는 회전 행렬을 구하여 값 갱신

    Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba) - estimator.g; //callback으로 들어온 가속도 센서 값도 보정

    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1); //두 값들의 평균을 구한다.

    tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc; // 평균 값을 통해 위치 갱신
    tmp_V = tmp_V + dt * un_acc;  // 평균 값을 통해 속도 갱신

    acc_0 = linear_acceleration; //이전 가속도와 자이로 값을 현재 값으로 변경
    gyr_0 = angular_velocity;
}

void update()
{
    TicToc t_predict;
    latest_time = current_time;
    tmp_P = estimator.Ps[WINDOW_SIZE]; //위치
    tmp_Q = estimator.Rs[WINDOW_SIZE]; // IMU 센서의 위치 및 방향을 나타내는 회전 행렬
    tmp_V = estimator.Vs[WINDOW_SIZE]; //속도 
    tmp_Ba = estimator.Bas[WINDOW_SIZE]; //가속도 bias 값
    tmp_Bg = estimator.Bgs[WINDOW_SIZE]; //자이로 bias 값
    acc_0 = estimator.acc_0; // 가속도 센서
    gyr_0 = estimator.gyr_0; // 자이로 센서

    queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = imu_buf; // imu 값이 저장된 버퍼를 복사
    for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop()) // 버퍼 값 순회
        predict(tmp_imu_buf.front()); // 버퍼에 들어온 센서 값을 통해 속도 및 위치 추정

}

std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>>
getMeasurements()
{
    std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements; // 센서 측정 값을 담는 배열

    while (true)
    {
        if (imu_buf.empty() || feature_buf.empty()) // imu 버퍼와 feature 버퍼가 비어 있으면 
            return measurements; // 배열을 반환하고 종료

        // 가장 최근에 들어온 센서 값의 time stamp 값이 버퍼의 가장 앞에 있는 이미지의 time stamp + imu 시간과 이미지 시간과의 차이 값보다 작거나 같으면
        //  => 최근에 수집된 센서 데이터가 버퍼의 첫 이미지보다 이전에 수집되었다면
        if (!(imu_buf.back()->header.stamp.toSec() > feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            //ROS_WARN("wait for imu, only should happen at the beginning");
            sum_of_wait++; // 센서 데이터 대기 횟수 증가
            return measurements; // 배열을 반환하고 종료
        }
        // 가장 먼저 들어온 센서 값의 time stamp 값이 버퍼의 가장 앞에 있는 이미지의 time stamp + imu 시간과 이미지 시간과의 차이 값보다 크거나 같으면
        //  => 가장 먼저 수집된 센서 데이터가 버퍼의 첫 이미지 이후에 수집되었다면
        if (!(imu_buf.front()->header.stamp.toSec() < feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            ROS_WARN("throw img, only should happen at the beginning");
            feature_buf.pop(); //feature 버퍼의 앞에 있는 이미지 정보를 버린다.
            continue; // 다음 loop로 이동
        }
        sensor_msgs::PointCloudConstPtr img_msg = feature_buf.front(); // feature 버퍼의 가장 앞에 있는 이미지 정보를 가져옴
        feature_buf.pop(); // 가장 앞에 있는 이미지 정보를 버퍼에서 버린다.

        std::vector<sensor_msgs::ImuConstPtr> IMUs; // imu 정보를 담을 배열
        while (imu_buf.front()->header.stamp.toSec() < img_msg->header.stamp.toSec() + estimator.td) // 센서의 time stamp가 이미지의 time stamp보다 작으면 반복
        {
            IMUs.emplace_back(imu_buf.front()); // 배열에 센서 데이터를 넣고
            imu_buf.pop(); //버퍼에서 정보 제거
        }
        IMUs.emplace_back(imu_buf.front()); // 반복 조건에서 빠져 나온 경우의 imu 데이터도 배열에 저장
        if (IMUs.empty()) // 배열이 비어 있다면
            ROS_WARN("no imu between two image");
        measurements.emplace_back(IMUs, img_msg); // 센서 측정 값을 담는 배열에 imu 배열과 image 정보를 넣는다.
    }
    return measurements; // 측정 배열 반환
}

void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    if (imu_msg->header.stamp.toSec() <= last_imu_t) //현재 들어온 imu의 timestamp가 마지막 timestamp보다 작거나 같으면
    {
        ROS_WARN("imu message in disorder!"); //경고 메시지 출력 후 함수 종료
        return;
    }
    last_imu_t = imu_msg->header.stamp.toSec(); //마지막 imu의 timestamp를 현재 들어온 timestamp로 초기화

    m_buf.lock(); //lock
    imu_buf.push(imu_msg); //버퍼에 메시지(imu 값)를 삽입
    m_buf.unlock(); //unlock
    con.notify_one(); //대기 중인 스레드 하나를 깨운다.

    last_imu_t = imu_msg->header.stamp.toSec(); //마지막 imu의 timestamp를 현재 들어온 timestamp로 초기화

    {
        std::lock_guard<std::mutex> lg(m_state); //블록 내에 lock을 건다.
        predict(imu_msg); // 현재 들어온 센서 값을 통해 속도 및 위치 추정
        std_msgs::Header header = imu_msg->header; // 현재 들어온 센서 값의 헤더 값(time stamp 등이 있음)을 저장
        header.frame_id = "world"; // 센서 데이터에는 frame 정보가 없으므로 임의로 world로 함
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR) // 센서 값이 NON_LINEAR 상태이면 (INITIAL 과NON_LINEAR 2가지 만 존재함)
            pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header); //센서 정보를 다른 노드로 보낸다.
    }
}


void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    if (!init_feature) // 처음 이미지 데이터가 들어오면 무시
    {
        //skip the first detected feature, which doesn't contain optical flow speed
        init_feature = 1;
        return;
    }
    m_buf.lock(); // lock
    feature_buf.push(feature_msg);
    m_buf.unlock(); // unlock
    con.notify_one(); // sleep하고 있는 스레드 중 하나를 깨운다.
}

void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true) // 다시 시작하라는 message 전달 시
    {
        ROS_WARN("restart the estimator!"); // 모든 값 초기화
        m_buf.lock();
        while(!feature_buf.empty())
            feature_buf.pop();
        while(!imu_buf.empty())
            imu_buf.pop();
        m_buf.unlock();
        m_estimator.lock();
        estimator.clearState();
        estimator.setParameter();
        m_estimator.unlock();
        current_time = -1;
        last_imu_t = 0;
    }
    return;
}

void relocalization_callback(const sensor_msgs::PointCloudConstPtr &points_msg)
{
    //printf("relocalization callback! \n");
    m_buf.lock();
    relo_buf.push(points_msg); // relocalization 버퍼에 massage 저장
    m_buf.unlock();
}

// thread: visual-inertial odometry
// VIO를 수행하는 스레드
void process()
{
    while (true)
    {
        std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements; // 센서 측정 값을 담는 배열 생성
        std::unique_lock<std::mutex> lk(m_buf); // 락 변수 선언
        con.wait(lk, [&]
                 {
            return (measurements = getMeasurements()).size() != 0;
                 }); // 측정 데이터가 들어올 때 까지 대기
        lk.unlock(); // unlock
        m_estimator.lock(); // lock
        for (auto &measurement : measurements) // 센서 측정 값 배열의 요소를 순회
        {
            auto img_msg = measurement.second; // image 정보를 가져온다.
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
            for (auto &imu_msg : measurement.first) // 해당 image 이전에 측정된 imu 데이터를 순회
            {
                double t = imu_msg->header.stamp.toSec(); // imu 데이터의 time stamp를 가져온다.
                double img_t = img_msg->header.stamp.toSec() + estimator.td; // 이미지 time stamp에 imu와의 시간 차이를 더해서 이미지의 time stamp로 함
                if (t <= img_t) // imu의 time stamp가 이미지의 time stamp보다 작거나 같으면
                { 
                    if (current_time < 0) // 현재 시간이 초기화되지 않았으면
                        current_time = t; // 현재 시간을 imu의 time stamp로 초기화
                    double dt = t - current_time; // dt를 imu의 time stamp와 현재 시간의 차이 값으로 함
                    ROS_ASSERT(dt >= 0); // dt가 0보다 작은 경우 프로그램이 멈추고 오류 메시지를 출력
                    current_time = t; // 현재 시간을 imu의 time stamp로 초기화
                    dx = imu_msg->linear_acceleration.x; // 가속도 센서의 x, y, z 값을 가져옴
                    dy = imu_msg->linear_acceleration.y;
                    dz = imu_msg->linear_acceleration.z;
                    rx = imu_msg->angular_velocity.x; // 자이로 센서의 x, y, z 값을 가져옴
                    ry = imu_msg->angular_velocity.y;
                    rz = imu_msg->angular_velocity.z;
                    estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz)); // dt와 가속도 센서 값, 자이로 센서 값을 넣고 속도, 위치, 회전 추정
                    //printf("imu: dt:%f a: %f %f %f w: %f %f %f\n",dt, dx, dy, dz, rx, ry, rz);

                }
                else // 큰 경우
                {
                    double dt_1 = img_t - current_time; // 이미지의 time stamp와 현재 시간의 차이를 dt_1로
                    double dt_2 = t - img_t; // imu time stamp와 image time stamp의 차이를 dt_2로
                    current_time = img_t; // 현재 시간 갱신
                    ROS_ASSERT(dt_1 >= 0); // dt_1이 0보다 작은 경우 종료
                    ROS_ASSERT(dt_2 >= 0); // dt_2가 0보다 작은 경우 종료
                    ROS_ASSERT(dt_1 + dt_2 > 0); // dt_1 + dt_2가 0 이하인 경우 종료
                    double w1 = dt_2 / (dt_1 + dt_2); //dt_1과 dt_2의 가중치를 구함
                    double w2 = dt_1 / (dt_1 + dt_2);
                    dx = w1 * dx + w2 * imu_msg->linear_acceleration.x; // 가중치를 통해 값을 혼합하여 자이로 가속도 값을 추정
                    dy = w1 * dy + w2 * imu_msg->linear_acceleration.y;
                    dz = w1 * dz + w2 * imu_msg->linear_acceleration.z;
                    rx = w1 * rx + w2 * imu_msg->angular_velocity.x;
                    ry = w1 * ry + w2 * imu_msg->angular_velocity.y;
                    rz = w1 * rz + w2 * imu_msg->angular_velocity.z;
                    estimator.processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz)); // dt와 가속도 센서 값, 자이로 센서 값을 넣고 속도, 위치, 회전 추정
                    //printf("dimu: dt:%f a: %f %f %f w: %f %f %f\n",dt_1, dx, dy, dz, rx, ry, rz);
                }
            }
            // set relocalization frame
            sensor_msgs::PointCloudConstPtr relo_msg = NULL; // relocalization massage 포인터 생성
            while (!relo_buf.empty()) // 버퍼에 내용이 있으면
            {
                relo_msg = relo_buf.front(); // 버퍼의 앞 부분을 꺼낸다.
                relo_buf.pop();
            }
            if (relo_msg != NULL) // massage에 내용이 있으면
            {
                vector<Vector3d> match_points; // 매칭점을 담을 배열 생성
                double frame_stamp = relo_msg->header.stamp.toSec(); // relocalization할 frame의 time stamp를 가져옴
                for (unsigned int i = 0; i < relo_msg->points.size(); i++) // 매칭 점의 개수 만큼 순회
                {
                    Vector3d u_v_id; 
                    u_v_id.x() = relo_msg->points[i].x; // 매칭점의 x, y, z 값을 저장
                    u_v_id.y() = relo_msg->points[i].y;
                    u_v_id.z() = relo_msg->points[i].z;
                    match_points.push_back(u_v_id); // 배열에 삽입
                }
                Vector3d relo_t(relo_msg->channels[0].values[0], relo_msg->channels[0].values[1], relo_msg->channels[0].values[2]); // keyframe에 대한 변환 행렬
                Quaterniond relo_q(relo_msg->channels[0].values[3], relo_msg->channels[0].values[4], relo_msg->channels[0].values[5], relo_msg->channels[0].values[6]); // key frame에 대한 회전 행렬
                Matrix3d relo_r = relo_q.toRotationMatrix(); // 쿼터니안을 Matrix로 변환
                int frame_index;
                frame_index = relo_msg->channels[0].values[7]; // frame의 index 저장
                estimator.setReloFrame(frame_stamp, frame_index, match_points, relo_t, relo_r); // relocalization할 frame 설정
            }

            ROS_DEBUG("processing vision data with stamp %f \n", img_msg->header.stamp.toSec());

            TicToc t_s;
            map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;
            for (unsigned int i = 0; i < img_msg->points.size(); i++)
            {
                int v = img_msg->channels[0].values[i] + 0.5;
                int feature_id = v / NUM_OF_CAM;
                int camera_id = v % NUM_OF_CAM;
                double x = img_msg->points[i].x;
                double y = img_msg->points[i].y;
                double z = img_msg->points[i].z;
                double p_u = img_msg->channels[1].values[i];
                double p_v = img_msg->channels[2].values[i];
                double velocity_x = img_msg->channels[3].values[i];
                double velocity_y = img_msg->channels[4].values[i];
                ROS_ASSERT(z == 1);
                Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                image[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
            }
            estimator.processImage(image, img_msg->header);

            double whole_t = t_s.toc();
            printStatistics(estimator, whole_t);
            std_msgs::Header header = img_msg->header;
            header.frame_id = "world";

            pubOdometry(estimator, header);
            pubKeyPoses(estimator, header);
            pubCameraPose(estimator, header);
            pubPointCloud(estimator, header);
            pubTF(estimator, header);
            pubKeyframe(estimator);
            if (relo_msg != NULL)
                pubRelocalization(estimator);
            //ROS_ERROR("end: %f, at %f", img_msg->header.stamp.toSec(), ros::Time::now().toSec());
        }
        m_estimator.unlock();
        m_buf.lock();
        m_state.lock();
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            update();
        m_state.unlock();
        m_buf.unlock();
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "vins_estimator"); //ROS node로 등록 (스레드를 만든다고 생각)
    ros::NodeHandle n("~"); //ROS 매개변수 서버의 로컬 namesapce로부터 매개 변수를 가져옴 (config파일 정보, vins_folder 정보)
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info); //ROS 로깅 레벨 설정 (프로그램의 실행 중에 발생하는 이벤트나 상태 정보를 기록, info는 일반적인 상태 정보)
    readParameters(n); //매개변수를 읽는다. (paramerters의 readParameters 함수 호출)
    estimator.setParameter(); //읽어온 매개변수를 estimator 객체에 저장
#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif
    ROS_WARN("waiting for image and imu..."); //이미지와 imu 데이터를 기다린다.

    registerPub(n); //메시지들을 publish하는 함수(utility/visualization.cpp)

    // /imu0 라는 토픽을 구독하고, 메시지 큐 크기는 2000으로 설정,새로운 메시지가 오면 img_callback함수를 호출(callback함수), TCP 지연 통신 최소화
    ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());
    // /feature_tracker/feature 라는 토픽을 구독하고, 메시지 큐 크기는 2000으로 설정,새로운 메시지가 오면 feature_callback함수를 호출(callback함수)
    ros::Subscriber sub_image = n.subscribe("/feature_tracker/feature", 2000, feature_callback);
    // /feature_tracker/restart 라는 토픽을 구독하고, 메시지 큐 크기는 2000으로 설정,새로운 메시지가 오면 restart_callback함수를 호출(callback함수)
    ros::Subscriber sub_restart = n.subscribe("/feature_tracker/restart", 2000, restart_callback);
    // /pose_graph/match_points 라는 토픽을 구독하고, 메시지 큐 크기는 2000으로 설정,새로운 메시지가 오면 relocalization_callback함수를 호출(callback함수)
    ros::Subscriber sub_relo_points = n.subscribe("/pose_graph/match_points", 2000, relocalization_callback);

    std::thread measurement_process{process}; //process 함수를 호출하는 thread 생성 및 실행
    ros::spin(); //프로그램이 종료될 때 까지 콜백 함수 처리를 기다림

    return 0;
}
