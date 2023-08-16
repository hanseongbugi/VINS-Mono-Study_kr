#include "initial_ex_rotation.h"

InitialEXRotation::InitialEXRotation(){
    frame_count = 0;
    Rc.push_back(Matrix3d::Identity());
    Rc_g.push_back(Matrix3d::Identity());
    Rimu.push_back(Matrix3d::Identity());
    ric = Matrix3d::Identity();
}

bool InitialEXRotation::CalibrationExRotation(vector<pair<Vector3d, Vector3d>> corres, Quaterniond delta_q_imu, Matrix3d &calib_ric_result)
{
    frame_count++; // frame count 증가
    Rc.push_back(solveRelativeR(corres)); // correspondent를 통해 Relative Roation 행렬 추정
    Rimu.push_back(delta_q_imu.toRotationMatrix()); // 쿼터니언인 imu 회전 행렬을 Matrix3d 객체로 변환해 배열에 삽입
    Rc_g.push_back(ric.inverse() * delta_q_imu * ric); // ric(현재 단위 행렬)의 역행렬에 delta_q(imu 회전 행렬)와 ric를 곱해 회전 행렬 보정 후 배열에 삽입

    Eigen::MatrixXd A(frame_count * 4, 4); //크기가 frame_count * 4x4 형태의 행렬 A를 생성
    A.setZero();
    int sum_ok = 0;
    for (int i = 1; i <= frame_count; i++) //frame_count 만큼 반복
    {
        Quaterniond r1(Rc[i]); // i번째 Relative Roation 행렬을 쿼터니안으로 변환
        Quaterniond r2(Rc_g[i]); // i번째 보정된 회전 행렬을 쿼터니안으로 변환

        double angular_distance = 180 / M_PI * r1.angularDistance(r2); //회전 행렬과 보정된 회전 행렬 사이의 각도 차이를 계산
        ROS_DEBUG(
            "%d %f", i, angular_distance);

        double huber = angular_distance > 5.0 ? 5.0 / angular_distance : 1.0; //각도 차이가 5.0보다 크면 huber를 1로 설정, 그렇지 않으면 5.0/각도 차이로 설정
        ++sum_ok;
        Matrix4d L, R; //크기가 4x4인 두 개의 행렬 L과 R을 생성

        double w = Quaterniond(Rc[i]).w(); // Relative Roation 행렬을 쿼터니언으로 변환 후 w값을 가져옴 (x, y, z는 벡터 값, w는 스칼라(roll) 값)
        Vector3d q = Quaterniond(Rc[i]).vec(); //Relative Roation 행렬을 쿼터니언으로 변환 후 x,y, z값을 가져옴
        L.block<3, 3>(0, 0) = w * Matrix3d::Identity() + Utility::skewSymmetric(q); // L 행렬의 3x3 값에 회전 값 저장
        L.block<3, 1>(0, 3) = q; // L 행렬의 3x1 값에 translation 값 저장
        L.block<1, 3>(3, 0) = -q.transpose(); // L 행렬의 1x3 값에 회전 행렬의 전치 행렬 저장
        L(3, 3) = w; // 회전 행렬의 w 값 저장

        Quaterniond R_ij(Rimu[i]); // imu 회전 행렬을 쿼터니언으로 변환
        w = R_ij.w(); //w값을 가져옴
        q = R_ij.vec(); //x,y, z값을 가져옴
        R.block<3, 3>(0, 0) = w * Matrix3d::Identity() - Utility::skewSymmetric(q); // R 행렬의 3x3 값에 회전 값 저장
        R.block<3, 1>(0, 3) = q; // R 행렬의 3x1 값에 translation 값 저장
        R.block<1, 3>(3, 0) = -q.transpose(); // R 행렬의 1x3 값에 회전 행렬의 전치 행렬 저장
        R(3, 3) = w; // 회전 행렬의 w 값 저장

        A.block<4, 4>((i - 1) * 4, 0) = huber * (L - R); //회전 행렬 L과 R의 차이에 huber 값 만큼 곱해서 A 행렬에 저장
    }

    JacobiSVD<MatrixXd> svd(A, ComputeFullU | ComputeFullV); //A 행렬을 분해
    Matrix<double, 4, 1> x = svd.matrixV().col(3); //회전 행렬의 보정값을 나타내는 4x1 행렬 저장
    Quaterniond estimated_R(x); //회전 행렬의 보정값을 나타내는 쿼터니언 객체 생성
    ric = estimated_R.toRotationMatrix().inverse(); //쿼터니언을 회전 행렬로 변환한 뒤, 그 역행렬을 구하여 ric에 저장
    //cout << svd.singularValues().transpose() << endl;
    //cout << ric << endl;
    Vector3d ric_cov;
    ric_cov = svd.singularValues().tail<3>(); //회전 행렬의 보정값이 얼마나 정확한지 값 가져옴

    // 현재까지의 프레임 수가 WINDOW_SIZE(10) 이상이고, 추정된 회전 행렬의 보정값 중 두 번째 값이 0.25보다 크다면
    if (frame_count >= WINDOW_SIZE && ric_cov(1) > 0.25)
    {
        calib_ric_result = ric; // 추정한 RIC 행렬을 calib_ric_result(인자로 받은 RIC 행렬)에 저장
        return true; // 추정 성공
    }
    else
        return false; //추정 실패
}

Matrix3d InitialEXRotation::solveRelativeR(const vector<pair<Vector3d, Vector3d>> &corres)
{
    if (corres.size() >= 9) // correspondent 배열의 크기가 9 이상이면 (corespondent가 9개 이상이면)
    {
        vector<cv::Point2f> ll, rr; // correspondent의 좌표를 담는 배열
        for (int i = 0; i < int(corres.size()); i++)
        {
            ll.push_back(cv::Point2f(corres[i].first(0), corres[i].first(1))); //왼쪽 좌표 (x, y) 저장
            rr.push_back(cv::Point2f(corres[i].second(0), corres[i].second(1))); // 오른쪽 좌표 (x, y) 저장
        }
        cv::Mat E = cv::findFundamentalMat(ll, rr); //좌표를 통해 Fundamental 행렬 계산
        cv::Mat_<double> R1, R2, t1, t2;
        decomposeE(E, R1, R2, t1, t2); //Fundamental 행렬을 두 개의 회전 행렬과 두 개의 변환 벡터로 분해

        if (determinant(R1) + 1.0 < 1e-09)  //첫 번째 회전 행렬 R1의 행렬식이 음수인 경우
        {
            E = -E; // Fundamental 행렬을 반전 시킨다.
            decomposeE(E, R1, R2, t1, t2); ////반전된 Fundamental 행렬을 두 개의 회전 행렬과 두 개의 변환 행렬로 분해
        }
        //두 개의 회전 행렬 R1, R2를 이용하여 triangulation을 수행한 후 ratio를 계산
        double ratio1 = max(testTriangulation(ll, rr, R1, t1), testTriangulation(ll, rr, R1, t2));
        double ratio2 = max(testTriangulation(ll, rr, R2, t1), testTriangulation(ll, rr, R2, t2));
        cv::Mat_<double> ans_R_cv = ratio1 > ratio2 ? R1 : R2; //두 회전 행렬 중 ratio가 높은 회전 행렬을 선택

        Matrix3d ans_R_eigen;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                ans_R_eigen(j, i) = ans_R_cv(i, j); //ans_R_cv의 데이터를 ans_R_eigen으로 복사
        return ans_R_eigen; //relative rotation 행렬 반환
    }
    return Matrix3d::Identity();
}

double InitialEXRotation::testTriangulation(const vector<cv::Point2f> &l,
                                          const vector<cv::Point2f> &r,
                                          cv::Mat_<double> R, cv::Mat_<double> t)
{
    cv::Mat pointcloud; // triangulation을 통해 추정된 점들을 저장할 행렬
    cv::Matx34f P = cv::Matx34f(1, 0, 0, 0,
                                0, 1, 0, 0,
                                0, 0, 1, 0); //평면 투영하는 변환행렬 초기화
    cv::Matx34f P1 = cv::Matx34f(R(0, 0), R(0, 1), R(0, 2), t(0),
                                 R(1, 0), R(1, 1), R(1, 2), t(1),
                                 R(2, 0), R(2, 1), R(2, 2), t(2)); // 회전 행렬과 변환 행렬을 통해 평면 투영 행렬 선언
    cv::triangulatePoints(P, P1, l, r, pointcloud); // triangulation을 통해 두개의 특징점 배열에 대응되는 점들을 추정, 추정된 점들은 pointcloud에 저장
    int front_count = 0; // 투영된 점들의 개수
    for (int i = 0; i < pointcloud.cols; i++) //추정된 점들을 순회
    {
        double normal_factor = pointcloud.col(i).at<float>(3); // 추정 결과로부터 normal factor를 가져옴

        cv::Mat_<double> p_3d_l = cv::Mat(P) * (pointcloud.col(i) / normal_factor); //첫 번째 투영 행렬을 이용하여 점들을 평면 투영
        cv::Mat_<double> p_3d_r = cv::Mat(P1) * (pointcloud.col(i) / normal_factor); //두 번째 투영 행렬을 이용하여 점들을 평면 투영
        if (p_3d_l(2) > 0 && p_3d_r(2) > 0) //점이 frame의 앞쪽에 위치해 있는 경우 (z 값이 양수인 경우)
            front_count++; //front_count 증가
    }
    ROS_DEBUG("MotionEstimator: %f", 1.0 * front_count / pointcloud.cols);
    return 1.0 * front_count / pointcloud.cols; //ratio 계산하고 반환
}

void InitialEXRotation::decomposeE(cv::Mat E,
                                 cv::Mat_<double> &R1, cv::Mat_<double> &R2,
                                 cv::Mat_<double> &t1, cv::Mat_<double> &t2)
{
    cv::SVD svd(E, cv::SVD::MODIFY_A); //Fundamental 행렬을 SVD를 통해 분해 (E = u * s * vt로 행렬을 분해)
    cv::Matx33d W(0, -1, 0,
                  1, 0, 0,
                  0, 0, 1); // 회전 행렬을 생성하기 위해 W와 Wt를 선언 (W는 Z축을 기준으로 90도 회전시키는 행렬, Wt는 W의 역행렬)
    cv::Matx33d Wt(0, 1, 0,
                   -1, 0, 0,
                   0, 0, 1);
    R1 = svd.u * cv::Mat(W) * svd.vt; //첫 번째 회전 행렬 R1을 계산
    R2 = svd.u * cv::Mat(Wt) * svd.vt; //두 번째 회전 행렬 R2를 계산
    t1 = svd.u.col(2); //첫 번째 변환 행렬 t1을 계산
    t2 = -svd.u.col(2); //두 번째 변환 행렬 t2를 계산
}
