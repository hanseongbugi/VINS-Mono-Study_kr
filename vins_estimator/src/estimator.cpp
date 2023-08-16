#include "estimator.h"

Estimator::Estimator(): f_manager{Rs}
{
    ROS_INFO("init begins"); 
    clearState(); // 변수 초기화
}

void Estimator::setParameter()
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = TIC[i]; //파라미터로 들어온 변환 행렬을 저장
        ric[i] = RIC[i];
    }
    f_manager.setRic(ric); //feature manager 객체에 ric 배열 전달
    ProjectionFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity(); //projection factor를 구하고 단위 행렬에 값을 저장
    ProjectionTdFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity(); 
    td = TD; //imu 시간과 이미지 시간과의 차이 값 저장
}

void Estimator::clearState()
{
    for (int i = 0; i < WINDOW_SIZE + 1; i++)
    {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr)
            delete pre_integrations[i];
        pre_integrations[i] = nullptr;
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }

    for (auto &it : all_image_frame)
    {
        if (it.second.pre_integration != nullptr)
        {
            delete it.second.pre_integration;
            it.second.pre_integration = nullptr;
        }
    }

    solver_flag = INITIAL;
    first_imu = false,
    sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    solver_flag = INITIAL;
    initial_timestamp = 0;
    all_image_frame.clear();
    td = TD;


    if (tmp_pre_integration != nullptr)
        delete tmp_pre_integration;
    if (last_marginalization_info != nullptr)
        delete last_marginalization_info;

    tmp_pre_integration = nullptr;
    last_marginalization_info = nullptr;
    last_marginalization_parameter_blocks.clear();

    f_manager.clearState();

    failure_occur = 0;
    relocalization_info = 0;

    drift_correct_r = Matrix3d::Identity();
    drift_correct_t = Vector3d::Zero();
}

void Estimator::processIMU(double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity)
{
    if (!first_imu) // 첫번째 imu 데이터를 받지 못한 경우
    {
        first_imu = true; // 첫번째 imu 데이터를 받았다고 flag 설정
        acc_0 = linear_acceleration; // 가속도 값을 인자로 받은 값으로 설정
        gyr_0 = angular_velocity; // 자이로 값을 인자로 받은 값으로 설정
    }

    if (!pre_integrations[frame_count]) // frame 번호에 해당하는 preIntegration 객체가 없는 경우
    {
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]}; // IntegrationBase 객체 생성 (Integration에 필요한 행렬 초기화)
    }
    if (frame_count != 0) // frame이 있는 경우
    {
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity); // dt와 가속도, 자이로 센서 값을 통해 preintegration 진행
        //if(solver_flag != NON_LINEAR)
            tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity); // 같은 값으로 한번더 integration 진행

        dt_buf[frame_count].push_back(dt); // dt 버퍼에 dt 값 저장
        linear_acceleration_buf[frame_count].push_back(linear_acceleration); // 가속도 버퍼에 가속도 값 저장 
        angular_velocity_buf[frame_count].push_back(angular_velocity); // 자이로 버퍼에 자이로 값 저장

        int j = frame_count;         
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g; //현재 프레임 번호에 대한 가속도 값
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j]; // 현재 프레임 번호에 대한 자이로 값
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix(); //회전 행렬 값 갱신 (자이로 값을 통해)
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g; // 겡신된 회전 행렬 값을 통해 현재 가속도 값 계산
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1); // 두 값의 평균을 구함
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc; //위치 갱신
        Vs[j] += dt * un_acc; // 속도 갱신
    }
    acc_0 = linear_acceleration; // 가속도 값을 인자로 받은 값으로 설정
    gyr_0 = angular_velocity; // 자이로 값을 인자로 받은 값으로 설정
}

void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const std_msgs::Header &header)
{
    ROS_DEBUG("new image coming ------------------------------------------");
    ROS_DEBUG("Adding feature points %lu", image.size());
    if (f_manager.addFeatureCheckParallax(frame_count, image, td)) // frame 수와 frame 정보, td를 넣고 특징 점 간 parallax 계산
        marginalization_flag = MARGIN_OLD; //marginalization flag를 OLD로 설정 (marginalization에서 이전 데이터를 사용)
    else
        marginalization_flag = MARGIN_SECOND_NEW; //marginalization flag를 SECOND_NEW로 설정 (해당 프레임을 키프레임으로 선택)

    ROS_DEBUG("this frame is--------------------%s", marginalization_flag ? "reject" : "accept");
    ROS_DEBUG("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    ROS_DEBUG("Solving %d", frame_count);
    ROS_DEBUG("number of feature: %d", f_manager.getFeatureCount()); // 특징점 개수 반환
    Headers[frame_count] = header; //frame_count번째 이미지 정보를 Headers 배열에 삽입

    ImageFrame imageframe(image, header.stamp.toSec()); //특징점 정보와 timeStamp를 통해 ImageFrame 객체 생성
    imageframe.pre_integration = tmp_pre_integration; // ImageFrame 객체에 preIntegration 정보 전달 (processIMU 함수를 통해 구한 integration 값)
    all_image_frame.insert(make_pair(header.stamp.toSec(), imageframe)); // 모든 이미지 프레임을 담는 map 객체에 time stamp와 이때 해당하는 ImageFrame 객체 저장
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]}; // frame count에 해당하는 bias를 통해 integration 진행

    if(ESTIMATE_EXTRINSIC == 2) // Imu와 camera간 변환 행렬을 추정해야 한다면
    {
        ROS_INFO("calibrating extrinsic param, rotation movement is needed");
        if (frame_count != 0) // 프레임이 들어왔다면
        {
            vector<pair<Vector3d, Vector3d>> corres = f_manager.getCorresponding(frame_count - 1, frame_count); // 이전 프레임 번호와 현재 프레임 번호를 통해 corespondent인 특징점을 구함
            Matrix3d calib_ric;
            // corespondent 배열과 imu 회전 행렬을 통해 RIC 행렬 추정
            if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric)) 
            {
                ROS_WARN("initial extrinsic rotation calib success");
                ROS_WARN_STREAM("initial extrinsic rotation: " << endl << calib_ric);
                ric[0] = calib_ric; // 추정한 RIC 복사
                RIC[0] = calib_ric;
                ESTIMATE_EXTRINSIC = 1; // 추정 frag 값을 1로 변경 (추정한 RIC를 최적화를 진행하기 위해) 
            }
        }
    }

    if (solver_flag == INITIAL) // solver flag가 초기화 상태라면
    {
        if (frame_count == WINDOW_SIZE) // frame count가 WINDOW_SIZE(10)과 같다면
        {
            bool result = false; // 결과 값 초기화
            // Imu와 camera간 변환 행렬을 추정해야하는 상태가 아니고 frame의 time stamp - 초기 time stamp값이 0.1보다 크면
            if( ESTIMATE_EXTRINSIC != 2 && (header.stamp.toSec() - initial_timestamp) > 0.1)
            {
               result = initialStructure(); // VINS-MONO에 필요한 자료 구조 초기화
               initial_timestamp = header.stamp.toSec(); // 초기 time stamp 값을 현재 frame의 time stamp로 초기화
            }
            if(result) // 초기화 성공 시
            {
                solver_flag = NON_LINEAR; // solver flag를 NON_LINEAR로 변경
                solveOdometry(); // triangulation 후 optimization 진행
                slideWindow(); // slideWindow 수행
                f_manager.removeFailures(); // solve_flag가 2인 feature를 배열에서 제거
                ROS_INFO("Initialization finish!");
                last_R = Rs[WINDOW_SIZE]; // 이전 R과 P를 갱신
                last_P = Ps[WINDOW_SIZE];
                last_R0 = Rs[0]; // 이전 R0와 P0를 갱신
                last_P0 = Ps[0];
                
            }
            else // 초기화 실패
                slideWindow(); // slideWindow 수행
        }
        else
            frame_count++; // frame count 증가
    }
    else // solver flag가 NON_LINEAR라면
    {
        TicToc t_solve; // 실행 시간 저장
        solveOdometry(); // triangulation 후 optimization 진행
        ROS_DEBUG("solver costs: %fms", t_solve.toc());

        if (failureDetection()) // failure 감지에 성공하였다면
        {
            ROS_WARN("failure detection!");
            failure_occur = 1; // failure가 발생하였다고 flag 설정
            clearState(); // 변수 초기화
            setParameter(); // 파라미터 초기화
            ROS_WARN("system reboot!");
            return;
        }

        TicToc t_margin; // 실행 시간 저장
        slideWindow(); // slideWindow 수행
        f_manager.removeFailures(); // solve_flag가 2인 feature를 배열에서 제거
        ROS_DEBUG("marginalization costs: %fms", t_margin.toc());
        // prepare output of VINS
        key_poses.clear(); // key_poses 배열의 모든 요소 제거
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.push_back(Ps[i]); // 카메라 pose(WINDOW_SIZE 만큼)를 key_pose로 한다

        last_R = Rs[WINDOW_SIZE]; // 이전 R과 P를 갱신
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0]; // 이전 R0와 P0를 갱신
    }
}
bool Estimator::initialStructure()
{
    TicToc t_sfm; // 실행 시간을 기록하는 객체 생성
    //check imu observibility
    {
        map<double, ImageFrame>::iterator frame_it; // map<double, ImageFrame> 형태인 map 객체의 iterator 생성
        Vector3d sum_g; // 가속도 값의 합
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++) // 지나온 모든 frame에 대해서 순회
        {
            double dt = frame_it->second.pre_integration->sum_dt; // 현재 iterator가 가리키는 integration객체의 dt 합을 dt로 선언
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt; // 현재 속도에 dt를 나눠 가속도 값을 구함
            sum_g += tmp_g; // 가속도 값의 합을 갱신
        }
        Vector3d aver_g;
        aver_g = sum_g * 1.0 / ((int)all_image_frame.size() - 1); // 가속도 값의 평균을 구한다,
        double var = 0; // 가속도 값들의 분산 값
        for (frame_it = all_image_frame.begin(), frame_it++; frame_it != all_image_frame.end(); frame_it++) // 지나온 모든 frame에 대해서 순회
        {
            double dt = frame_it->second.pre_integration->sum_dt; // 현재 iterator가 가리키는 integration객체의 dt 합을 dt로 선언
            Vector3d tmp_g = frame_it->second.pre_integration->delta_v / dt; // 현재 속도에 dt를 나눠 가속도 값을 구함

            // 모든 ImageFrame 객체들의 가속도 값과 평균 가속도의 차이를 제곱하여 모두 더한 결과가 var에 저장된다.
            var += (tmp_g - aver_g).transpose() * (tmp_g - aver_g); 
            //cout << "frame g " << tmp_g.transpose() << endl;
        }
        var = sqrt(var / ((int)all_image_frame.size() - 1));  // 분산 값을 frame의 개수로 나누고 표준편차로 변환
        //ROS_WARN("IMU variation %f!", var);
        if(var < 0.25) // 표준편차가 0.25보다 작으면
        {
            ROS_INFO("IMU excitation not enouth!"); 
            //return false;
        }
    }
    // global sfm
    Quaterniond Q[frame_count + 1]; //frame count 만큼 카메라의 회전을 나타내는 배열을 선언
    Vector3d T[frame_count + 1]; //frame count 만큼 카메라의 위치를 나타내는 배열을 선언
    map<int, Vector3d> sfm_tracked_points; // 추정한 점의 좌표를 저장하는 map 객체 선언
    vector<SFMFeature> sfm_f; // SFMFeature 객체를 저장하는 배열 선언
    for (auto &it_per_id : f_manager.feature) // FeatureManager 객체의 feature 배열을 순회
    {
        int imu_j = it_per_id.start_frame - 1; //iterator가 가리키는 특징점의 시작 프레임 인덱스
        SFMFeature tmp_feature; //SFMFeature 객체 생성
        tmp_feature.state = false; // 상태를 false로 초기화
        tmp_feature.id = it_per_id.feature_id; // id를 feature_id로 초기화
        for (auto &it_per_frame : it_per_id.feature_per_frame) //iterator가 가리키는 feature_per_frame 배열 순회
        {
            imu_j++; //특징점의 시작 프레임 인덱스 증가
            Vector3d pts_j = it_per_frame.point; // 특징점 좌표를 받는다.
            tmp_feature.observation.push_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()})); // SFMFeature의 observation 배열에 인덱스와 특징점 좌표 저장
        }
        sfm_f.push_back(tmp_feature); // SFMFeature 객체를 배열에 삽입
    } 
    Matrix3d relative_R; //상대 회전 행렬
    Vector3d relative_T; //상대 위치 행렬
    int l; // 두 프레임 사이의 상대 frame 인덱스
    if (!relativePose(relative_R, relative_T, l)) // 상대 회전 행렬과 위치 행렬을 구하지 못하였다면
    {
        ROS_INFO("Not enough features or parallax; Move device around");
        return false; // 실패
    }
    GlobalSFM sfm; //GlobalSFM 객체를 생성
    if(!sfm.construct(frame_count + 1, Q, T, l,
              relative_R, relative_T,
              sfm_f, sfm_tracked_points)) // PnP, triangulate, Bundle Adjustment를 통해 point를 추정하여 sfm_tracked_points에 저장, 이때 3개의 연산 중 하나라도 실패 시
    {
        ROS_DEBUG("global SFM failed!");
        marginalization_flag = MARGIN_OLD; //marginalization flag를 OLD로 설정 (marginalization에서 이전 데이터를 사용)
        return false; // 실패
    }

    //solve pnp for all frame
    map<double, ImageFrame>::iterator frame_it; // <double, ImageFrame>형태인 map 객체의 iterator 생성
    map<int, Vector3d>::iterator it; // <int, Vector3d>형태인 map 객체의 iterator 생성
    frame_it = all_image_frame.begin( ); // 모든 이미지 프레임을 담고있는 map 객체의 첫 요소를 가리키게 한다.
    for (int i = 0; frame_it != all_image_frame.end( ); frame_it++) // iterator를 이용해 모든 이미지 프레임을 담고있는 map 객체 순회
    {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        if((frame_it->first) == Headers[i].stamp.toSec()) // 현재 가리키는 frame의 time stamp가 Headers 배열에 존재한다면
        {
            frame_it->second.is_key_frame = true; // 현재 가리키는 frame을 keyFrame으로 한다.
            frame_it->second.R = Q[i].toRotationMatrix() * RIC[0].transpose(); // 카메라 Rotation 행렬과 IMU Rotation 행렬을 곱하여 Frame Rotation 행렬로 초기화
            frame_it->second.T = T[i]; // 카메라 Translation 행렬을 Frame Translation 행렬로 초기화
            i++; // i 값 증가
            continue; // 아래 무시 (PnP 진행 안함)
        }
        if((frame_it->first) > Headers[i].stamp.toSec())// 현재 가리키는 frame의 time stamp가 Headers 배열의 i번째 time stamp보다 크다면
        {
            i++; // i 값 증가
        }
        Matrix3d R_inital = (Q[i].inverse()).toRotationMatrix(); // 카메라 Rotation 행렬의 역행렬을 쿼터니안에서 Matrix3d 객체로 변환하여 R_inital행렬 생성
        Vector3d P_inital = - R_inital * T[i]; // R_inital 행렬에 음수를 곱하고 카메라 Translation 행렬을 곱하여 P_inital 행렬 생성
        cv::eigen2cv(R_inital, tmp_r); // R_inital 행렬을 opencv의 Mat 객체로 변경
        cv::Rodrigues(tmp_r, rvec); // 회전 행렬을 Rodrigues 변환
        cv::eigen2cv(P_inital, t); // P_inital 행렬을 opencv의 Mat 객체로 변경

        frame_it->second.is_key_frame = false; // 현재 가리키는 frame은 keyFrame이 아니라고 설정
        vector<cv::Point3f> pts_3_vector; 
        vector<cv::Point2f> pts_2_vector;
        for (auto &id_pts : frame_it->second.points) // feature의 개수만큼 순회
        {
            int feature_id = id_pts.first; // feature id를 가져옴
            for (auto &i_p : id_pts.second) //feature의 position을 가져오기 위해 배열 순회
            {
                it = sfm_tracked_points.find(feature_id); 
                if(it != sfm_tracked_points.end()) // Global SFM을 통해 추정한 point 중 feature_id와 같은 것이 있다면 
                {
                    Vector3d world_pts = it->second; // 추정된 Point 좌표를 얻고
                    cv::Point3f pts_3(world_pts(0), world_pts(1), world_pts(2)); // opencv의 Point3f 객체 생성
                    pts_3_vector.push_back(pts_3); // 배열에 삽입
                    Vector2d img_pts = i_p.second.head<2>(); // 특징점의 이미지 상의 좌표를 얻고
                    cv::Point2f pts_2(img_pts(0), img_pts(1));
                    pts_2_vector.push_back(pts_2); // 배열에 삽입
                }
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1); // 카메라 파라미터 K 행렬 생성   
        if(pts_3_vector.size() < 6) // 추정된 Point 개수가 6보다 작을 경우
        {
            cout << "pts_3_vector size " << pts_3_vector.size() << endl;
            ROS_DEBUG("Not enough points for solve pnp !");
            return false;
        }
        if (! cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1)) // PnP에 실패할 경우
        {
            ROS_DEBUG("solve pnp fail!");
            return false;
        }
        cv::Rodrigues(rvec, r); // Rodrigues 변환된 Rotation 행렬을 원래 형태로 변환
        MatrixXd R_pnp,tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp); // opencv의 Mat을 Eigen의 Matrix 형식으로 변환
        R_pnp = tmp_R_pnp.transpose(); // Roation 행렬 전치
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp); // opencv의 Mat을 Eigen의 Matrix 형식으로 변환
        T_pnp = R_pnp * (-T_pnp); // Translation = 회전 행렬 * Translation 행렬에 음수를 곱한 값
        frame_it->second.R = R_pnp * RIC[0].transpose(); // Rotation 행렬에 Imu와 카메라간 Rotation 행렬을 전치하여 곱하여 프레임의 Rotation update
        frame_it->second.T = T_pnp; // Translation Update
    }
    if (visualInitialAlign()) // 카메라 및 IMU 상태 조정 업데이트에 성공한다면 
        return true;
    else // 실패 시
    {
        ROS_INFO("misalign visual structure with IMU");
        return false;
    }

}

bool Estimator::visualInitialAlign()
{
    TicToc t_g;
    VectorXd x;
    //solve scale
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x); //IMU 데이터와 frame 데이터를 이용하여 중력 벡터 및 gyro bias 보정
    if(!result) // 보정 실패 시
    {
        ROS_DEBUG("solve g failed!");
        return false;
    }

    // change state
    for (int i = 0; i <= frame_count; i++) // frame_count 만큼 순회
    {
        Matrix3d Ri = all_image_frame[Headers[i].stamp.toSec()].R; // 모든 frame에서 i에 해당하는 이미지 frame의 rotation 행렬을 가져옴
        Vector3d Pi = all_image_frame[Headers[i].stamp.toSec()].T; // 모든 frame에서 i에 해당하는 이미지 frame의 translation 행렬을 가져옴
        Ps[i] = Pi; // 가져온 Translation 행렬을 카메라 위치로
        Rs[i] = Ri; // rotation 행렬을 카메라 회전으로 
        all_image_frame[Headers[i].stamp.toSec()].is_key_frame = true; // 해당 이미지 프레임을 키 프레임으로 설정
    }

    VectorXd dep = f_manager.getDepthVector(); // feature의 depth 배열을 가져옴
    for (int i = 0; i < dep.size(); i++) // 배열의 크기만큼 순회
        dep[i] = -1; // depth를 -1로 초기화
    f_manager.clearDepth(dep); // depth 값 갱신

    //triangulat on cam pose , no tic
    Vector3d TIC_TMP[NUM_OF_CAM]; // 카메라 개수만큼 TIC_TMP 행렬 배열 생성
    for(int i = 0; i < NUM_OF_CAM; i++)
        TIC_TMP[i].setZero(); // TIC_TMP 행렬을 0행렬로 초기화
    ric[0] = RIC[0]; // 0번째에 대한 RIC 행렬을 가져옴
    f_manager.setRic(ric); // FeatureManager 객체의 RIC 갱신
    f_manager.triangulate(Ps, &(TIC_TMP[0]), &(RIC[0])); // 위치와 TIC, RIC를 통해 triangulation 진행

    double s = (x.tail<1>())(0); // scale을 가져온다. 
    for (int i = 0; i <= WINDOW_SIZE; i++) // WINDOW 크기만큼 순회
    {
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]); // bias를 통해 IMU 데이터 재보정
    }
    for (int i = frame_count; i >= 0; i--) // frame_count를 역으로 순회
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]); // 카메라 위치를 scale 값을 통해 보정
    int kv = -1;
    map<double, ImageFrame>::iterator frame_i;
    for (frame_i = all_image_frame.begin(); frame_i != all_image_frame.end(); frame_i++) // 모든 frame에 대해 순회
    {
        if(frame_i->second.is_key_frame) // keyFrame인 경우
        {
            kv++; // index증가
            Vs[kv] = frame_i->second.R * x.segment<3>(kv * 3); // 속도를 keyFrame의 Rotation 행렬을 x에 있는 속도 정보와 곱해서 속도 값으로 갱신 
        }
    }
    for (auto &it_per_id : f_manager.feature) // 특징점 배열 순회
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size(); // FeaturePerFrame 객체의 크기를 used_num으로 설정
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2)) // used_num이 2보다 작고, start_frame의 index가 WINDOW_SIZE - 2(8)보다 크면
            continue; // 무시
        it_per_id.estimated_depth *= s; // depth를 scale 크기만큼 곱해서 설정
    }

    Matrix3d R0 = Utility::g2R(g); // 중력 가속도 행렬을 Rotation 행렬로 변환
    double yaw = Utility::R2ypr(R0 * Rs[0]).x(); // R0 * 카메라 회전 값을 yaw, ptich, row로 변환 후 yaw 값을 가져옴
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0; // -yaw값을 Rotation 행렬로 변환하고 R0를 곱해 R0 갱신
    g = R0 * g; // Rotation 행렬과 중력 가속도를 곱해 중력 가속도 보정
    //Matrix3d rot_diff = R0 * Rs[0].transpose();
    Matrix3d rot_diff = R0; // Rotation 행렬을 가중치 행렬로 설정
    for (int i = 0; i <= frame_count; i++) // frame_count 만큼 순회
    {
        Ps[i] = rot_diff * Ps[i]; // 카메라 위치, 회전, 속도에 가중치를 곱해 보정
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
    }
    ROS_DEBUG_STREAM("g0     " << g.transpose());
    ROS_DEBUG_STREAM("my R0  " << Utility::R2ypr(Rs[0]).transpose()); 

    return true;
}

bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l)
{
    // find previous frame which contians enough correspondance and parallex with newest frame
    for (int i = 0; i < WINDOW_SIZE; i++) // WINDOW_SIZE(10) 만큼 반복
    {
        vector<pair<Vector3d, Vector3d>> corres; 
        corres = f_manager.getCorresponding(i, WINDOW_SIZE); //FeatureManager 객체에서 i번째 frame과 WINDOW 사이 correspondent를 가져온다.
        if (corres.size() > 20) //대응하는 특징점 개수가 20을 넘는 경우
        {
            double sum_parallax = 0; // parallax의 합 초기화
            double average_parallax; // parallax의 평균
            for (int j = 0; j < int(corres.size()); j++) // correspondent 배열의 크기만큼 순회
            {
                Vector2d pts_0(corres[j].first(0), corres[j].first(1)); // correspondent의 첫번째 point 좌표를 가져옴
                Vector2d pts_1(corres[j].second(0), corres[j].second(1)); // correspondent의 두번째 point 좌표를 가져옴
                double parallax = (pts_0 - pts_1).norm(); // 두 개의 벡터의 차이 값을 normalization하여(거리를 구하여) pararllax를 구한다.
                sum_parallax = sum_parallax + parallax; // parallax의 합 갱신

            }
            average_parallax = 1.0 * sum_parallax / int(corres.size()); // parallax 평균 값 계산
            // 평균 parallax가 30보다 크고 MotionEstimator 객체(initial/solve_5pts.cpp)의 solveRelativeRT를 통해 상대적인 Rotation 행렬과 Translation 행렬를 추정 했다면
            if(average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, relative_R, relative_T))
            {
                l = i; // i값을 l 변수에 저장
                ROS_DEBUG("average_parallax %f choose l %d and newest frame to triangulate the whole structure", average_parallax * 460, l);
                return true; //추정이 성공
            }
        }
    }
    return false; // 추정 실패
}

void Estimator::solveOdometry()
{
    if (frame_count < WINDOW_SIZE) // frame_count가 WINDOW_SIZE보다 작으면 함수를 종료
        return;
    if (solver_flag == NON_LINEAR) // solver_flag가 NON_LINEAR이면
    {
        TicToc t_tri; 
        f_manager.triangulate(Ps, tic, ric); // 카메라 위치, Tic, Ric를 통해 triangulation 진행
        ROS_DEBUG("triangulation costs %f", t_tri.toc()); // tritangulation 실행 시간 출력
        optimization(); // optimization 진행
    }
}

void Estimator::vector2double()
{
    for (int i = 0; i <= WINDOW_SIZE; i++)
    {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        para_SpeedBias[i][0] = Vs[i].x();
        para_SpeedBias[i][1] = Vs[i].y();
        para_SpeedBias[i][2] = Vs[i].z();

        para_SpeedBias[i][3] = Bas[i].x();
        para_SpeedBias[i][4] = Bas[i].y();
        para_SpeedBias[i][5] = Bas[i].z();

        para_SpeedBias[i][6] = Bgs[i].x();
        para_SpeedBias[i][7] = Bgs[i].y();
        para_SpeedBias[i][8] = Bgs[i].z();
    }
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);
    if (ESTIMATE_TD)
        para_Td[0][0] = td;
}

void Estimator::double2vector()
{
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];

    if (failure_occur)
    {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = 0;
    }
    Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                      para_Pose[0][3],
                                                      para_Pose[0][4],
                                                      para_Pose[0][5]).toRotationMatrix());
    double y_diff = origin_R0.x() - origin_R00.x();
    //TODO
    Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
    if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0)
    {
        ROS_DEBUG("euler singular point!");
        rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                       para_Pose[0][3],
                                       para_Pose[0][4],
                                       para_Pose[0][5]).toRotationMatrix().transpose();
    }

    for (int i = 0; i <= WINDOW_SIZE; i++)
    {

        Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
        
        Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                para_Pose[i][1] - para_Pose[0][1],
                                para_Pose[i][2] - para_Pose[0][2]) + origin_P0;

        Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                    para_SpeedBias[i][1],
                                    para_SpeedBias[i][2]);

        Bas[i] = Vector3d(para_SpeedBias[i][3],
                          para_SpeedBias[i][4],
                          para_SpeedBias[i][5]);

        Bgs[i] = Vector3d(para_SpeedBias[i][6],
                          para_SpeedBias[i][7],
                          para_SpeedBias[i][8]);
    }

    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        tic[i] = Vector3d(para_Ex_Pose[i][0],
                          para_Ex_Pose[i][1],
                          para_Ex_Pose[i][2]);
        ric[i] = Quaterniond(para_Ex_Pose[i][6],
                             para_Ex_Pose[i][3],
                             para_Ex_Pose[i][4],
                             para_Ex_Pose[i][5]).toRotationMatrix();
    }

    VectorXd dep = f_manager.getDepthVector();
    for (int i = 0; i < f_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    f_manager.setDepth(dep);
    if (ESTIMATE_TD)
        td = para_Td[0][0];

    // relative info between two loop frame
    if(relocalization_info)
    { 
        Matrix3d relo_r;
        Vector3d relo_t;
        relo_r = rot_diff * Quaterniond(relo_Pose[6], relo_Pose[3], relo_Pose[4], relo_Pose[5]).normalized().toRotationMatrix();
        relo_t = rot_diff * Vector3d(relo_Pose[0] - para_Pose[0][0],
                                     relo_Pose[1] - para_Pose[0][1],
                                     relo_Pose[2] - para_Pose[0][2]) + origin_P0;
        double drift_correct_yaw;
        drift_correct_yaw = Utility::R2ypr(prev_relo_r).x() - Utility::R2ypr(relo_r).x();
        drift_correct_r = Utility::ypr2R(Vector3d(drift_correct_yaw, 0, 0));
        drift_correct_t = prev_relo_t - drift_correct_r * relo_t;   
        relo_relative_t = relo_r.transpose() * (Ps[relo_frame_local_index] - relo_t);
        relo_relative_q = relo_r.transpose() * Rs[relo_frame_local_index];
        relo_relative_yaw = Utility::normalizeAngle(Utility::R2ypr(Rs[relo_frame_local_index]).x() - Utility::R2ypr(relo_r).x());
        //cout << "vins relo " << endl;
        //cout << "vins relative_t " << relo_relative_t.transpose() << endl;
        //cout << "vins relative_yaw " <<relo_relative_yaw << endl;
        relocalization_info = 0;    

    }
}

bool Estimator::failureDetection()
{
    if (f_manager.last_track_num < 2) // 마지막 프레임에서 feature 수가 2개 미만일 경우
    {
        ROS_INFO(" little feature %d", f_manager.last_track_num);
        //return true;
    }
    if (Bas[WINDOW_SIZE].norm() > 2.5) // acc bias의 norm이 2.5보다 크거나
    {
        ROS_INFO(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
        return true; // failure 감지 성공
    }
    if (Bgs[WINDOW_SIZE].norm() > 1.0) // gyro bias의 norm아 1.0보다 큰 경우
    {
        ROS_INFO(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
        return true; // failure 감지 성공
    }
    /*
    if (tic(0) > 1)
    {
        ROS_INFO(" big extri param estimation %d", tic(0) > 1);
        return true;
    }
    */
    Vector3d tmp_P = Ps[WINDOW_SIZE]; // WINDOW_SIZE에 해당하는 camera pose 저장
    if ((tmp_P - last_P).norm() > 5) // 이전 camera pose와의 차이의 norm이 5보다 크면
    {
        ROS_INFO(" big translation");
        return true;  // failure 감지 성공
    }
    if (abs(tmp_P.z() - last_P.z()) > 1) // 이전 camera pose와의 z값 차이가 1보다 크면
    {
        ROS_INFO(" big z translation");
        return true; // failure 감지 성공
    }
    Matrix3d tmp_R = Rs[WINDOW_SIZE]; // WINDOW_SIZE에 해당하는 tranlation 행렬을 가져옴
    Matrix3d delta_R = tmp_R.transpose() * last_R; // 이전 rotation 행렬과 곱 연산 진행
    Quaterniond delta_Q(delta_R); // 쿼터니안으로 변환
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0; // delta_Q를 통해 angle 계산
    if (delta_angle > 50) // 각이 50보다 크면
    {
        ROS_INFO(" big delta_angle ");
        //return true;
    }
    return false; // failure 감지 실패
}


void Estimator::optimization()
{
    ceres::Problem problem; // Optimization을 위해 Ceres Solver의 Problem 객체를 생성
    ceres::LossFunction *loss_function; 
    //loss_function = new ceres::HuberLoss(1.0);
    loss_function = new ceres::CauchyLoss(1.0); // 최적화 가중치 정의
    for (int i = 0; i < WINDOW_SIZE + 1; i++) // WINDOW_SIZE만큼 loop
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization(); // 최적화에 사용할 파라미터 생성
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization); // 카메라 pose를 Optimizatio 객체에 삽입
        problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEEDBIAS); // bias를 Optimizatio 객체에 삽입
    }
    for (int i = 0; i < NUM_OF_CAM; i++) // 카메라 수 만큼 loop
    {
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization(); // 최적화에 사용할 파라미터 생성
        problem.AddParameterBlock(para_Ex_Pose[i], SIZE_POSE, local_parameterization); // Imu 관련 행렬을 Optimizatio 객체에 삽입
        if (!ESTIMATE_EXTRINSIC) // ESTIMATE_EXTRINSIC가 0인 경우 (카메라 IMU관련 행렬을 추정하지 않는 경우)
        {
            ROS_DEBUG("fix extinsic param");
            problem.SetParameterBlockConstant(para_Ex_Pose[i]); // Imu 관련 행렬을 Optimizatio 객체에 삽입 (이 과정으로 Tic, Ric는 불변하게 된다)
        }
        else
            ROS_DEBUG("estimate extinsic param");
    }
    if (ESTIMATE_TD) // Imu와 Camera 간 time stamp가 일치하지 않는 경우
    {
        problem.AddParameterBlock(para_Td[0], 1); // config.yaml에서 정의한 td 값을 Optimizatio 객체에 삽입
        //problem.SetParameterBlockConstant(para_Td[0]);
    }

    TicToc t_whole, t_prepare;
    vector2double(); // 카메라 pose, Ric, Tic를 para_Pose, para_Ex_Pose 배열에 삽입

    if (last_marginalization_info) // 이전 marginalization이 존재한다면
    {
        // construct new marginlization_factor
        MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info); // 새로운 marginalization 객체 생성 (이때 이전 정보 사용)
        problem.AddResidualBlock(marginalization_factor, NULL,
                                 last_marginalization_parameter_blocks); // Optimizatio 객체에 삽입
    }

    for (int i = 0; i < WINDOW_SIZE; i++) // WINDOW_SIZE만큼 loop
    {
        int j = i + 1;
        if (pre_integrations[j]->sum_dt > 10.0) // dt가 10보다 큰 경우 무시
            continue;
        IMUFactor* imu_factor = new IMUFactor(pre_integrations[j]); // Integration 정보를 통해 IMUFactor 생성 (pre_integration 정보를 가지는 객체)
        problem.AddResidualBlock(imu_factor, NULL, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]); // Optimizatio 객체에 삽입
    }
    int f_m_cnt = 0;
    int feature_index = -1;
    for (auto &it_per_id : f_manager.feature) // feature 배열 loop
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size(); // feature를 얻는데 사용한 frame 수를 feature_per_frame 배열의 크기로 설정
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2)) // feature를 얻는데 사용된 frame 수가 2보다 작거나 startFrame이 8 이상이면 무시
            continue;
 
        ++feature_index; // feature index 증가

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1; // imu_i는 현재 feature의 start_frame 번호로하고 imu_j는 start_frame의 이전 index로 함
        
        Vector3d pts_i = it_per_id.feature_per_frame[0].point; // feature_per_frame을 통해 0번째 feature의 위치를 가져옴

        for (auto &it_per_frame : it_per_id.feature_per_frame) // feature_per_frame배열 순회
        {
            imu_j++; // imu_j index 증가
            if (imu_i == imu_j) // 두 index가 같은 경우 무시
            {
                continue;
            }
            Vector3d pts_j = it_per_frame.point; // feature의 위치를 가져옴
            if (ESTIMATE_TD) // Imu와 Camera 간 time stamp가 일치하지 않는 경우
            {
                    ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                     it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                                                     it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y()); // feature위치와 velocity, td를 통해 Projection 객체 생성
                    problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]); // Optimizatio 객체에 삽입
                    /*
                    double **para = new double *[5];
                    para[0] = para_Pose[imu_i];
                    para[1] = para_Pose[imu_j];
                    para[2] = para_Ex_Pose[0];
                    para[3] = para_Feature[feature_index];
                    para[4] = para_Td[0];
                    f_td->check(para);
                    */
            }
            else
            {
                ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j); // feature위치를 통해 Projection 객체 생성
                problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]); // Optimizatio 객체에 삽입
            }
            f_m_cnt++;
        }
    }

    ROS_DEBUG("visual measurement count: %d", f_m_cnt);
    ROS_DEBUG("prepare for ceres: %f", t_prepare.toc());

    if(relocalization_info) // relocalization_info가 0이 아닌 경우
    {
        //printf("set relocalization factor! \n");
        ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization(); // Pose Local 파라미터를 설정
        problem.AddParameterBlock(relo_Pose, SIZE_POSE, local_parameterization); // Optimizatio 객체에 삽입
        int retrive_feature_index = 0;
        int feature_index = -1;
        for (auto &it_per_id : f_manager.feature) // feature 배얄 loop
        {
            it_per_id.used_num = it_per_id.feature_per_frame.size(); // feature를 얻는데 사용한 frame 수를 feature_per_frame 배열의 크기로 설정
            if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2)) // feature를 얻는데 사용된 frame 수가 2보다 작거나 startFrame이 8 이상이면 무시
                continue;
            ++feature_index;
            int start = it_per_id.start_frame; // 현재 feature의 start_frame index 저장
            if(start <= relo_frame_local_index) // start_frame index가 relocalization frame의 index보다 작거나 같은 경우
            {   
                while((int)match_points[retrive_feature_index].z() < it_per_id.feature_id) // 매칭점 배열의 z 값이 feature_id보다 작으면
                {
                    retrive_feature_index++; // 다음 index 번호로
                }
                if((int)match_points[retrive_feature_index].z() == it_per_id.feature_id) //// 매칭점 배열의 z 값이 feature_id와 같다면
                {
                    Vector3d pts_j = Vector3d(match_points[retrive_feature_index].x(), match_points[retrive_feature_index].y(), 1.0); // 매칭점의 x, y값을 통해 pts_j 생성
                    Vector3d pts_i = it_per_id.feature_per_frame[0].point; // 현재 feature의 첫 번째 프레임에서의 point 정보를 이용하여 pts_i 생성
                    
                    ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j); // pts_i와 pts_j를 통해 Projection 파라미터 생성
                    problem.AddResidualBlock(f, loss_function, para_Pose[start], relo_Pose, para_Ex_Pose[0], para_Feature[feature_index]); // Optimizatio 객체에 삽입
                    retrive_feature_index++; // 다음 index 번호로
                }     
            }
        }

    }

    ceres::Solver::Options options; // Optimization의 Option 객체 생성

    options.linear_solver_type = ceres::DENSE_SCHUR; // linear_solver_type을 Dense_schur로 설정
    //options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS; //  최대 반복 횟수를 설정 (8회)
    //options.use_explicit_schur_complement = true;
    //options.minimizer_progress_to_stdout = true;
    //options.use_nonmonotonic_steps = true;
    if (marginalization_flag == MARGIN_OLD) // marginalization_flag가 MARGIN_OLD인 경우 
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0; // solver 실행 시간을 SOLVER_TIME(0.04) * 4/5 로 설정
    else
        options.max_solver_time_in_seconds = SOLVER_TIME; // solver 실행 시간을 SOLVER_TIME(0.04)로 설정
    TicToc t_solver; // 현재 실행 시간 저장
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary); // Optimization 문제를 풀고, 결과를 summary에 저장
    //cout << summary.BriefReport() << endl;
    ROS_DEBUG("Iterations : %d", static_cast<int>(summary.iterations.size()));
    ROS_DEBUG("solver costs: %f", t_solver.toc());

    double2vector(); // para_Pose, para_Ex_Pose 배열에 있는 값을 카메라 pose, Ric, Tic로 변환

    TicToc t_whole_marginalization; // 현재 실행 시간 저장
    if (marginalization_flag == MARGIN_OLD) // marginalization_flag가 MARGIN_OLD인 경우 
    {
        MarginalizationInfo *marginalization_info = new MarginalizationInfo(); // marginalizationInfo 객체 생성
        vector2double(); // 카메라 pose, Ric, Tic를 para_Pose, para_Ex_Pose 배열에 삽입

        if (last_marginalization_info) // last_marginalization_info가 존재하는 경우
        {
            vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++) // last_marginalization_parameter_block 배열 크기만큼 loop
            {
                if (last_marginalization_parameter_blocks[i] == para_Pose[0] ||
                    last_marginalization_parameter_blocks[i] == para_SpeedBias[0]) // last_marginalization_parameter_block이 카메라 pose와 같거나 bias와 같다면
                    drop_set.push_back(i); // 현재 index를 배열에 삽입
            }
            // construct new marginlization_factor
            MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info); // last_marginlization_info를 통해 MarginalizationFactor 객체 생성
            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set); // Factor를 통해 ResidualBlockInfo 객체 생성

            marginalization_info->addResidualBlockInfo(residual_block_info); // marginalization_info 객체에 ResidualBlockInfo 객체를 삽입
        }

        {
            if (pre_integrations[1]->sum_dt < 10.0) // pre_integrations 배열의 첫 번째 요소의 dt가 10 미만인 경우
            {
                IMUFactor* imu_factor = new IMUFactor(pre_integrations[1]); // 첫 번째 요소를 통해 Factor 객체 생성
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, NULL,
                                                                           vector<double *>{para_Pose[0], para_SpeedBias[0], para_Pose[1], para_SpeedBias[1]},
                                                                           vector<int>{0, 1}); // Factor를 통해 ResidualBlockInfo 객체 생성
                marginalization_info->addResidualBlockInfo(residual_block_info); // marginalization_info 객체에 ResidualBlockInfo 객체를 삽입
            }
        }

        {
            int feature_index = -1;
            for (auto &it_per_id : f_manager.feature) // feature 배열 순회
            {
                it_per_id.used_num = it_per_id.feature_per_frame.size(); // feature를 얻는데 사용한 frame 수를 feature_per_frame 배열의 크기로 설정
                if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2)) // feature를 얻는데 사용된 frame 수가 2보다 작거나 startFrame이 8 이상이면 무시
                    continue;

                ++feature_index; // index 증가

                int imu_i = it_per_id.start_frame, imu_j = imu_i - 1; // imu_i는 현재 feature의 start_frame 번호로하고 imu_j는 start_frame의 이전 index로 함
                if (imu_i != 0) // start_frame이 0이 아닌 경우
                    continue; // 무시

                Vector3d pts_i = it_per_id.feature_per_frame[0].point; // feature_per_frame의 0번째 요소의 point를 가져옴

                for (auto &it_per_frame : it_per_id.feature_per_frame) // feature_per_frame 배열 순회
                {
                    imu_j++; 
                    if (imu_i == imu_j) // imu_i와 imu_j가 같은 경우 무시
                        continue;

                    Vector3d pts_j = it_per_frame.point; // feature_per_frame 배열 요소의 point를 가져옴
                    if (ESTIMATE_TD) // Imu와 Camera 간 time stamp가 일치하지 않는 경우
                    {
                        ProjectionTdFactor *f_td = new ProjectionTdFactor(pts_i, pts_j, it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                                          it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                                                          it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y()); // feature위치와 velocity, td를 통해 Projection 객체 생성
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f_td, loss_function,
                                                                                        vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index], para_Td[0]},
                                                                                        vector<int>{0, 3}); // Factor, 카메라 pose와 Ric, Tic, Feature 정보를 통해 ResidualBlockInfo 객체 생성
                        marginalization_info->addResidualBlockInfo(residual_block_info); // marginalization_info 객체에 ResidualBlockInfo 객체 삽입
                    }
                    else
                    {
                        ProjectionFactor *f = new ProjectionFactor(pts_i, pts_j); // feature위치를 통해 Projection 객체 생성
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function,
                                                                                       vector<double *>{para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0], para_Feature[feature_index]},
                                                                                       vector<int>{0, 3}); // Factor, 카메라 pose와 Ric, Tic, Feature 정보를 통해 ResidualBlockInfo 객체 생성
                        marginalization_info->addResidualBlockInfo(residual_block_info); // marginalization_info 객체에 ResidualBlockInfo 객체 삽입
                    }
                }
            }
        }

        TicToc t_pre_margin; // 현재 시간 저장
        marginalization_info->preMarginalize(); // marginalization_info를 통해 preMarginalization 수행
        ROS_DEBUG("pre marginalization %f ms", t_pre_margin.toc());
        
        TicToc t_margin;
        marginalization_info->marginalize(); // marginalization_info를 통해 marginalization 수행
        ROS_DEBUG("marginalization %f ms", t_margin.toc());

        std::unordered_map<long, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++) // WINDOW_SIZE만큼 반복
        {
            addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1]; // camera pose와 bias를 shift에 삽입
            addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
        }
        for (int i = 0; i < NUM_OF_CAM; i++) // 카메라 수 만큼 loop
            addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i];  // Ric Tic를 shift에 삽입
        if (ESTIMATE_TD) // Imu와 Camera 간 time stamp가 일치하지 않는 경우
        {
            addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0]; // td 값 shift에 삽입
        }
        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift); // 구성한 shift를 통해 marginalization_info에서 parameter_blocks를 가져옴

        if (last_marginalization_info) // last_marginlization_info가 존재하는 경우
            delete last_marginalization_info; // 객체 삭제
        last_marginalization_info = marginalization_info; // last_marginlization_info 갱신
        last_marginalization_parameter_blocks = parameter_blocks; //last_marginalization_parameter_blocks 갱신
        
    }
    else
    {
        // last_marginalization_info가 존재하고, last_marginalization_parameter_blocks에 para_Pose의 WINDOW_SIZE - 1번째 요소가 포함되는 경우
        if (last_marginalization_info &&
            std::count(std::begin(last_marginalization_parameter_blocks), std::end(last_marginalization_parameter_blocks), para_Pose[WINDOW_SIZE - 1]))
        {

            MarginalizationInfo *marginalization_info = new MarginalizationInfo(); // marginalizationInfo 객체 생성
            vector2double(); // 카메라 pose, Ric, Tic를 para_Pose, para_Ex_Pose 배열에 삽입
            if (last_marginalization_info) // last_marginalization_info가 존재하는 경우
            {
                vector<int> drop_set; 
                for (int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); i++) // last_marginalization_parameter_block 배열 크기만큼 loop
                {
                    ROS_ASSERT(last_marginalization_parameter_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]); // 두 배열의 값이 같으면 프로그램 종료
                    if (last_marginalization_parameter_blocks[i] == para_Pose[WINDOW_SIZE - 1]) // camera pose가 같으면
                        drop_set.push_back(i); // 해당 index 삽입
                }
                // construct new marginlization_factor
                MarginalizationFactor *marginalization_factor = new MarginalizationFactor(last_marginalization_info);  // last_marginlization_info를 통해 MarginalizationFactor 객체 생성
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, NULL,
                                                                               last_marginalization_parameter_blocks,
                                                                               drop_set);  // Factor를 통해 ResidualBlockInfo 객체 생성

                marginalization_info->addResidualBlockInfo(residual_block_info); // marginalization_info 객체에 ResidualBlockInfo 객체를 삽입
            }

            TicToc t_pre_margin; // 현재 시간 저장
            ROS_DEBUG("begin marginalization");
            marginalization_info->preMarginalize();  // marginalization_info를 통해 preMarginalization 수행
            ROS_DEBUG("end pre marginalization, %f ms", t_pre_margin.toc());
             
            TicToc t_margin; // 현재 시간 저장
            ROS_DEBUG("begin marginalization");
            marginalization_info->marginalize(); // marginalization_info를 통해 marginalization 수행
            ROS_DEBUG("end marginalization, %f ms", t_margin.toc());
            
            std::unordered_map<long, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++) // WINDOW_SIZE만큼 반복
            {
                if (i == WINDOW_SIZE - 1) // i가 WINDOW_SIZE - 1(9)인 경우 무시
                    continue;
                else if (i == WINDOW_SIZE) // // i가 WINDOW_SIZE(10)인 경우
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i - 1];  // camera pose와 bias를 shift에 삽입 (i일 때 pose를 key로 하고, i-1일 때 pose를 value로)
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i - 1];
                }
                else // 일반적인 경우
                {
                    addr_shift[reinterpret_cast<long>(para_Pose[i])] = para_Pose[i];  // camera pose와 bias를 shift에 삽입 (i일 때 pose를 key로 하고, i일 때 pose를 value로)
                    addr_shift[reinterpret_cast<long>(para_SpeedBias[i])] = para_SpeedBias[i];
                }
            }
            for (int i = 0; i < NUM_OF_CAM; i++)
                addr_shift[reinterpret_cast<long>(para_Ex_Pose[i])] = para_Ex_Pose[i]; // Ric Tic를 shift에 삽입
            if (ESTIMATE_TD)
            {
                addr_shift[reinterpret_cast<long>(para_Td[0])] = para_Td[0];  // td 값 shift에 삽입
            }
            
            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift); // 구성한 shift를 통해 marginalization_info에서 parameter_blocks를 가져옴
            if (last_marginalization_info) // last_marginlization_info가 존재하는 경우
                delete last_marginalization_info; // 객체 삭제
            last_marginalization_info = marginalization_info; // last_marginlization_info 갱신
            last_marginalization_parameter_blocks = parameter_blocks; //last_marginalization_parameter_blocks 갱신
             
        }
    }
    ROS_DEBUG("whole marginalization costs: %f", t_whole_marginalization.toc());
    
    ROS_DEBUG("whole time for ceres: %f", t_whole.toc()); // 총 소요 시간 출력
}

void Estimator::slideWindow()
{
    TicToc t_margin; // 현재 시간 저장
    if (marginalization_flag == MARGIN_OLD) // marginlization_flag가 MARGIN_OLD인 경우
    {
        double t_0 = Headers[0].stamp.toSec(); // time stamp를 t_0에 저장
        back_R0 = Rs[0]; // 0번째 카메라 rotation 행렬 저장
        back_P0 = Ps[0]; // 0번째 카메라 pose 저장
        if (frame_count == WINDOW_SIZE) // frame_count가 WINDOW_SIZE와 같다면
        {
            for (int i = 0; i < WINDOW_SIZE; i++) // WINDOW_SIZE 만큼 순회
            {
                Rs[i].swap(Rs[i + 1]); // Rotation 배열의 i번째 요소와 i + 1번째 요소의 순서를 바꿈

                std::swap(pre_integrations[i], pre_integrations[i + 1]); // integration 배열의 i번째 요소와 i + 1번째 요소의 순서를 바꿈

                dt_buf[i].swap(dt_buf[i + 1]); // dt 버퍼의 i번째 요소와 i + 1번째 요소의 순서를 바꿈
                linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]); // 가속도 배열의 i번째 요소와 i + 1번째 요소의 순서를 바꿈
                angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]); // gyro 배열의 i번째 요소와 i + 1번째 요소의 순서를 바꿈

                Headers[i] = Headers[i + 1]; // Headers 배열의 i번째 요소와 i + 1번째 요소의 순서를 바꿈
                Ps[i].swap(Ps[i + 1]); // 카메라 pose 배열의 i번째 요소와 i + 1번째 요소의 순서를 바꿈
                Vs[i].swap(Vs[i + 1]); // 속도 배열의 i번째 요소와 i + 1번째 요소의 순서를 바꿈
                Bas[i].swap(Bas[i + 1]); // 가속도 bias 배열의 i번째 요소와 i + 1번째 요소의 순서를 바꿈
                Bgs[i].swap(Bgs[i + 1]); // 자이로 bias 배열의 i번째 요소와 i + 1번째 요소의 순서를 바꿈
            }
            Headers[WINDOW_SIZE] = Headers[WINDOW_SIZE - 1]; // WINDOW_SIZE 번째 요소를 WINDOW_SIZE - 1 번째 요소로 복사
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];
            Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
            Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

            delete pre_integrations[WINDOW_SIZE];  // WINDOW_SIZE 번째 객체 제거
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]}; // WINDOW_SIZE 번째 객체를 새롭게 생성해서 삽입

            dt_buf[WINDOW_SIZE].clear(); // 버퍼 초기화
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            if (true || solver_flag == INITIAL) // solver_flag가 INITIAL인 경우 (하지만 true가 있어 항상 통과하는 if문)
            {
                map<double, ImageFrame>::iterator it_0; // <doble, ImageFrame> 형태의 map을 가리키는 iterator 생성
                it_0 = all_image_frame.find(t_0); // 모든 이미지 frame 배열에서 t_0를 key로 하는 value를 찾는다
                delete it_0->second.pre_integration; // pre_integration 객체를 제거함
                it_0->second.pre_integration = nullptr; // null을 가지게 한다.
 
                for (map<double, ImageFrame>::iterator it = all_image_frame.begin(); it != it_0; ++it) // 모든 이미지 프레임 배열 순회
                {
                    if (it->second.pre_integration) // pre_integration 객체가 있다면
                        delete it->second.pre_integration; // 객체 제거
                    it->second.pre_integration = NULL; // null을 가지게한다.
                }

                all_image_frame.erase(all_image_frame.begin(), it_0); // 모든 이미지 프레임 배열에서 it_0까지 있는 요소를 제거함
                all_image_frame.erase(t_0); // t_0를 key로 하는 value를 제거

            }
            slideWindowOld(); // slide window를 통해 오래된 데이터 및 상태를 정리
        }
    }
    else
    {
        if (frame_count == WINDOW_SIZE) // frame_count가 WINDOW_SIZE와 같다면
        {
            for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++)
            {
                double tmp_dt = dt_buf[frame_count][i]; // dt 버퍼에서 현재 프레임의 dt를 가져옴
                Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i]; // 가속도 버퍼에서 현재 프레임의 가속도를 가져옴
                Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i]; // 자이로 버퍼에서 현재 프레임의 자이로를 가져옴

                pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity); // 현재 정보를 이전 정보로 갱신

                dt_buf[frame_count - 1].push_back(tmp_dt);
                linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
                angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
            }

            Headers[frame_count - 1] = Headers[frame_count]; // 현재 Header 값을 이전 Header 값으로 갱신
            Ps[frame_count - 1] = Ps[frame_count]; // 카메라 위치, 속도, 회전, bias를 갱신
            Vs[frame_count - 1] = Vs[frame_count];
            Rs[frame_count - 1] = Rs[frame_count];
            Bas[frame_count - 1] = Bas[frame_count];
            Bgs[frame_count - 1] = Bgs[frame_count];

            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]}; // WINDOW_SIZE 번째 pre_integrations 객체 변경

            dt_buf[WINDOW_SIZE].clear(); // WINDOW_SIZE번째 버퍼 삭제
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            slideWindowNew(); // 새로운 데이터 및 상태를 정리
        }
    }
}

// real marginalization is removed in solve_ceres()
void Estimator::slideWindowNew()
{
    sum_of_front++; // sum_of_back 1 증가
    f_manager.removeFront(frame_count); // feature 배열 정리
}
// real marginalization is removed in solve_ceres()
void Estimator::slideWindowOld()
{
    sum_of_back++; // sum_of_back 1 증가

    bool shift_depth = solver_flag == NON_LINEAR ? true : false; // solver_flag 값이 NON_LINEAR인 경우 shift_depth를 true로 설정
    if (shift_depth) // shift_depth가 true인 경우
    {
        Matrix3d R0, R1;
        Vector3d P0, P1;
        R0 = back_R0 * ric[0]; // 이전 카메라 rotation 행랼에 Ric를 곱함
        R1 = Rs[0] * ric[0]; // 0번째 카메라 rotation 행렬에 Ric를 곱함
        P0 = back_P0 + back_R0 * tic[0]; // 이전 카메라 rotation 행렬과 Tic를 곱하고 카메라 pose를 더함
        P1 = Ps[0] + Rs[0] * tic[0]; // 0번째 카메라 rotation 행렬과 Tic를 곱하고 0번째 카메라 pose를 더함
        f_manager.removeBackShiftDepth(R0, P0, R1, P1); // feature 배열의 depth를 재조정 및 feature 배열 정리
    }
    else
        f_manager.removeBack(); // feature 배열 정리
}

void Estimator::setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d> &_match_points, Vector3d _relo_t, Matrix3d _relo_r)
{
    relo_frame_stamp = _frame_stamp; // relocalization할 time stamp 저장
    relo_frame_index = _frame_index; // relocalization할 frame index 저장
    match_points.clear(); // 매칭 점 행렬 초기화
    match_points = _match_points; // 매칭점 행렬을 인자로 들어온 값으로
    prev_relo_t = _relo_t; // 변환 행렬
    prev_relo_r = _relo_r; // 회전 행렬
    for(int i = 0; i < WINDOW_SIZE; i++)
    {
        if(relo_frame_stamp == Headers[i].stamp.toSec()) // relocalization할 frame의 time stamp가 지금까지 지나온 frame 중 하나라면 
        {
            relo_frame_local_index = i; // 지나온 frame의 index를 저장
            relocalization_info = 1; // relocalization할 수 있도록 frag 설정
            for (int j = 0; j < SIZE_POSE; j++)
                relo_Pose[j] = para_Pose[i][j]; // relocalization할 frame의 index에 해당하는 위치와 회전 행렬 값을 저장
        }
    }
}

