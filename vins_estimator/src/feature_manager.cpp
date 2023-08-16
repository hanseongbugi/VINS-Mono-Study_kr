#include "feature_manager.h"

int FeaturePerId::endFrame()
{
    return start_frame + feature_per_frame.size() - 1;
}

FeatureManager::FeatureManager(Matrix3d _Rs[])
    : Rs(_Rs)
{
    for (int i = 0; i < NUM_OF_CAM; i++)
        ric[i].setIdentity();
}

void FeatureManager::setRic(Matrix3d _ric[])
{
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ric[i] = _ric[i];
    }
}

void FeatureManager::clearState()
{
    feature.clear();
}

int FeatureManager::getFeatureCount()
{
    int cnt = 0;
    for (auto &it : feature)
    {

        it.used_num = it.feature_per_frame.size();

        if (it.used_num >= 2 && it.start_frame < WINDOW_SIZE - 2)
        {
            cnt++;
        }
    }
    return cnt;
}


bool FeatureManager::addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td)
{
    ROS_DEBUG("input feature: %d", (int)image.size());
    ROS_DEBUG("num of feature: %d", getFeatureCount());
    double parallax_sum = 0; // parallax 거리의 합
    int parallax_num = 0; // parallax를 계산 수
    last_track_num = 0; // tracking 수
    for (auto &id_pts : image)
    {
        FeaturePerFrame f_per_fra(id_pts.second[0].second, td); // 특정 frame에 관한 특정점 정보를 담는 객체 생성 (0번째 카메라에 대한 특징점 행렬과 td 사용)

        int feature_id = id_pts.first; // 특징점 id 값 저장
        auto it = find_if(feature.begin(), feature.end(), [feature_id](const FeaturePerId &it)
                          {
            return it.feature_id == feature_id;
                          }); // feature_id와 일치하는 FeaturePerId객체를 찾는다. (it은 feature 배열에 있는 FeaturePerId객체)

        if (it == feature.end()) // feature_id와 같은 FeaturePerId 객체가 없다면
        {
            feature.push_back(FeaturePerId(feature_id, frame_count)); // feature 배열에 feature_id에 해당하는 FeaturePerId객체를 생성하고 삽입
            feature.back().feature_per_frame.push_back(f_per_fra); // 방금 삽입한 객체의 feature_per_frame 배열에 FeaturePerFrame 객체 삽입
        }
        else if (it->feature_id == feature_id) // feature_id와 같은 객체를 찾았다면
        {
            it->feature_per_frame.push_back(f_per_fra); // 찾은 객체의 feature_per_frame 객체에 FeaturePerFrame 객체 삽입
            last_track_num++; // tracking 수 증가
        }
    }

    if (frame_count < 2 || last_track_num < 20) // frame count가 2보다 작거나 tracking 수가 20보다 작으면
        return true; // true를 반환하고 함수 종료

    for (auto &it_per_id : feature) // FeaturePerId 객체를 담고있는 feature 배열 순회
    {
        // feature 배열에 있는 객체의 start_frame이 frame_count - 2 보다 작거나 같고, start_framer과 feature_per_frame 배열의 크기가 frame count -1보다 크거나 같으면
        //  => 배열에 있는 객체의 frame이 이전 frame이면
        if (it_per_id.start_frame <= frame_count - 2 &&
            it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1) 
        {
            parallax_sum += compensatedParallax2(it_per_id, frame_count); // FeaturePerId 객체와 frame_count를 통해 Parallax 추정하고 거리를 반환 받음
            parallax_num++; // parallax 계산 수 증가
        }
    }

    if (parallax_num == 0) // parallax를 구하지 못하였다면
    {
        return true; // true를 반환하고 함수 종료
    }
    else
    {
        ROS_DEBUG("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
        ROS_DEBUG("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH);
        return parallax_sum / parallax_num >= MIN_PARALLAX; // 거리 합 / 거리 계산 수(거리 평균)가 최소 Parallax 값(10, config파일에 있는 정보)보다 크거나 같으면 true
    }
}

void FeatureManager::debugShow()
{
    ROS_DEBUG("debug show");
    for (auto &it : feature)
    {
        ROS_ASSERT(it.feature_per_frame.size() != 0);
        ROS_ASSERT(it.start_frame >= 0);
        ROS_ASSERT(it.used_num >= 0);

        ROS_DEBUG("%d,%d,%d ", it.feature_id, it.used_num, it.start_frame);
        int sum = 0;
        for (auto &j : it.feature_per_frame)
        {
            ROS_DEBUG("%d,", int(j.is_used));
            sum += j.is_used;
            printf("(%lf,%lf) ",j.point(0), j.point(1));
        }
        ROS_ASSERT(it.used_num == sum);
    }
}

vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(int frame_count_l, int frame_count_r)
{
    vector<pair<Vector3d, Vector3d>> corres; //corespondent인 특징점 위치를 저장하는 배열
    for (auto &it : feature) // FeaturePerId 객체를 담고있는 feature 배열 순회
    {
        // 배열에 있는 frame이 이전 프레임 번호보다 작거나 같고 객체의 마지막 프레임 번호가 현재 프레임번호보다 크거나 같으면
        // 값들이 사이에 존재한다면
        if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r) 
        {
            Vector3d a = Vector3d::Zero(), b = Vector3d::Zero(); // vector 초기화
            int idx_l = frame_count_l - it.start_frame; // left 프레임 idx를 이전 프레임 번호 - 객체의 프레임 id
            int idx_r = frame_count_r - it.start_frame; // right 프레임 idx를 현재 프레임 번호 - 객체의 프레임 id

            a = it.feature_per_frame[idx_l].point; // id에 해당하는 특징점을 가져온다

            b = it.feature_per_frame[idx_r].point; // id에 해당하는 특징점을 가져온다
            
            corres.push_back(make_pair(a, b)); // 2개의 특징점을 corespondent 배열에 삽입
        }
    }
    return corres;
}

void FeatureManager::setDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        it_per_id.estimated_depth = 1.0 / x(++feature_index);
        //ROS_INFO("feature id %d , start_frame %d, depth %f ", it_per_id->feature_id, it_per_id-> start_frame, it_per_id->estimated_depth);
        if (it_per_id.estimated_depth < 0)
        {
            it_per_id.solve_flag = 2;
        }
        else
            it_per_id.solve_flag = 1;
    }
}

void FeatureManager::removeFailures()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        if (it->solve_flag == 2)
            feature.erase(it);
    }
}

void FeatureManager::clearDepth(const VectorXd &x)
{
    int feature_index = -1; // feature index 초기화
    for (auto &it_per_id : feature) // feature 배열 순회
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size(); // feature_per_frame 배열의 크기를 used_num(feature가 사용된 frame 수)에 저장
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2)) // used_num의 크기가 2 이상이고 시작 프레임이 WINDOWSIZE-2(8)보다 작으면
            continue; // 무시
        it_per_id.estimated_depth = 1.0 / x(++feature_index);  // feature의 depth를 인자로 들어온 x 배열의 값으로 변경 ( x 배열의 depth는 역수 이기에 다시 역수를 취함)
    }
}

VectorXd FeatureManager::getDepthVector()
{
    VectorXd dep_vec(getFeatureCount()); // feature의 개수만큼 크기를 가진 Nx1 행렬 생성
    int feature_index = -1; // feature index 초기화
    for (auto &it_per_id : feature) // feature 배열 순회
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size(); // feature_per_frame 배열의 크기를 used_num(feature가 사용된 frame 수)에 저장
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2)) // used_num의 크기가 2 이상이고 시작 프레임이 WINDOWSIZE-2(8)보다 작으면
            continue; // 무시
#if 1
        dep_vec(++feature_index) = 1. / it_per_id.estimated_depth; // feature의 depth의 역수를 배열 저장
#else
        dep_vec(++feature_index) = it_per_id->estimated_depth; // 컴파일되지 않는 부분임 ( if 1은 항상 참이기 때문 )
#endif
    }
    return dep_vec; // dep_vec 배열 반환
}

void FeatureManager::triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[])
{
    for (auto &it_per_id : feature)
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size();
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        if (it_per_id.estimated_depth > 0)
            continue;
        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        ROS_ASSERT(NUM_OF_CAM == 1);
        Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4);
        int svd_idx = 0;

        Eigen::Matrix<double, 3, 4> P0;
        Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
        Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];
        P0.leftCols<3>() = Eigen::Matrix3d::Identity();
        P0.rightCols<1>() = Eigen::Vector3d::Zero();

        for (auto &it_per_frame : it_per_id.feature_per_frame)
        {
            imu_j++;

            Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];
            Eigen::Matrix3d R1 = Rs[imu_j] * ric[0];
            Eigen::Vector3d t = R0.transpose() * (t1 - t0);
            Eigen::Matrix3d R = R0.transpose() * R1;
            Eigen::Matrix<double, 3, 4> P;
            P.leftCols<3>() = R.transpose();
            P.rightCols<1>() = -R.transpose() * t;
            Eigen::Vector3d f = it_per_frame.point.normalized();
            svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
            svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

            if (imu_i == imu_j)
                continue;
        }
        ROS_ASSERT(svd_idx == svd_A.rows());
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        double svd_method = svd_V[2] / svd_V[3];
        //it_per_id->estimated_depth = -b / A;
        //it_per_id->estimated_depth = svd_V[2] / svd_V[3];

        it_per_id.estimated_depth = svd_method;
        //it_per_id->estimated_depth = INIT_DEPTH;

        if (it_per_id.estimated_depth < 0.1)
        {
            it_per_id.estimated_depth = INIT_DEPTH;
        }

    }
}

void FeatureManager::removeOutlier()
{
    ROS_BREAK();
    int i = -1;
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        i += it->used_num != 0;
        if (it->used_num != 0 && it->is_outlier == true)
        {
            feature.erase(it);
        }
    }
}

void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P)
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            Eigen::Vector3d uv_i = it->feature_per_frame[0].point;  
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() < 2)
            {
                feature.erase(it);
                continue;
            }
            else
            {
                Eigen::Vector3d pts_i = uv_i * it->estimated_depth;
                Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P;
                Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P);
                double dep_j = pts_j(2);
                if (dep_j > 0)
                    it->estimated_depth = dep_j;
                else
                    it->estimated_depth = INIT_DEPTH;
            }
        }
        // remove tracking-lost feature after marginalize
        /*
        if (it->endFrame() < WINDOW_SIZE - 1)
        {
            feature.erase(it);
        }
        */
    }
}

void FeatureManager::removeBack()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            it->feature_per_frame.erase(it->feature_per_frame.begin());
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

void FeatureManager::removeFront(int frame_count)
{
    for (auto it = feature.begin(), it_next = feature.begin(); it != feature.end(); it = it_next)
    {
        it_next++;

        if (it->start_frame == frame_count)
        {
            it->start_frame--;
        }
        else
        {
            int j = WINDOW_SIZE - 1 - it->start_frame;
            if (it->endFrame() < frame_count - 1)
                continue;
            it->feature_per_frame.erase(it->feature_per_frame.begin() + j);
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id, int frame_count)
{
    //check the second last frame is keyframe or not
    //parallax betwwen seconde last frame and third last frame
    const FeaturePerFrame &frame_i = it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame]; // FeaturePerId객체의 start_frame 기준 세번째 마지막 프레임 정보
    const FeaturePerFrame &frame_j = it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame]; // FeaturePerId객체의 start_frame 기준 두번째 마지막 프레임 정보

    double ans = 0; // parallax 값
    Vector3d p_j = frame_j.point; // 두번째 마지막 프레임의 특징점 위치

    double u_j = p_j(0); // 특징점의 x 값
    double v_j = p_j(1); // 특징점의 y 값

    Vector3d p_i = frame_i.point; // 세번째 마지막 프레임의 특징점 위치
    Vector3d p_i_comp; 

    //int r_i = frame_count - 2;
    //int r_j = frame_count - 1;
    //p_i_comp = ric[camera_id_j].transpose() * Rs[r_j].transpose() * Rs[r_i] * ric[camera_id_i] * p_i;
    p_i_comp = p_i; // p_i_comp에 p_i 값 복사
    double dep_i = p_i(2); // 세번째 마지막 프레임의 z 값 (depth)
    double u_i = p_i(0) / dep_i; // x값에 depth를 나누어 특징점의 2D 위치를 구함
    double v_i = p_i(1) / dep_i; // y값에 depth를 나누어 특징점의 2D 위치를 구함
    double du = u_i - u_j, dv = v_i - v_j; // 두 프레임 간의 2D 위치 차이를 계산

    double dep_i_comp = p_i_comp(2); // 세번째 마지막 프레임의 z 값 (depth)
    double u_i_comp = p_i_comp(0) / dep_i_comp; // x값에 depth를 나누어 특징점의 2D 위치를 구함
    double v_i_comp = p_i_comp(1) / dep_i_comp; // y값에 depth를 나누어 특징점의 2D 위치를 구함
    double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j; // 두 프레임 간의 추정된 2D 위치 차이를 계산

    ans = max(ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp))); //두 위치 차이의 유클리드 거리 중 큰 값을 ans에 저장

    return ans; // 거리 반환
}