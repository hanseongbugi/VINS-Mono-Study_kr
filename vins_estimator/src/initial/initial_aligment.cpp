#include "initial_alignment.h"

void solveGyroscopeBias(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs)
{
    Matrix3d A; // Bias 보정에 사용되는 행렬 생성
    Vector3d b;
    Vector3d delta_bg; // gyro bias의 보정 값
    A.setZero(); // A및 b 행렬을 0행렬로 초기화
    b.setZero();
    map<double, ImageFrame>::iterator frame_i; // all_image_frame에 접근하기 위한 iterator 생성
    map<double, ImageFrame>::iterator frame_j;
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++) // 모든 이미치 frame에 대해 순회
    {
        frame_j = next(frame_i); // frame_i가 가리키는 frame의 다음 frame을 가져옴
        MatrixXd tmp_A(3, 3); // 3x3 행렬 생성
        tmp_A.setZero(); // 0으로 초기화
        VectorXd tmp_b(3); // 3x1 행렬 생성
        tmp_b.setZero(); // 0으로 초기화
        Eigen::Quaterniond q_ij(frame_i->second.R.transpose() * frame_j->second.R); // 각 프레임의 Rotation 행렬을 통해 두 프레임 간의 Rotation을 나타내는 쿼터니안을 계산
        tmp_A = frame_j->second.pre_integration->jacobian.template block<3, 3>(O_R, O_BG); // pre_integration의 jacobian 행렬의 일부 사용
        tmp_b = 2 * (frame_j->second.pre_integration->delta_q.inverse() * q_ij).vec(); //imu 회전 행렬과 프레인간 Rotation을 통해 tmp_b 연산
        A += tmp_A.transpose() * tmp_A; // tmp_A의 전치와 tmp_A를 곱해 A 행렬에 누적
        b += tmp_A.transpose() * tmp_b; // tmp_A의 전치와 tmp_b를 곱해 B 행렬에 누적

    }
    delta_bg = A.ldlt().solve(b); // A * delta_bg = b 라는 선형식을 풀어(LDLT 분해) bias 보정 값 연산
    ROS_WARN_STREAM("gyroscope bias initial calibration " << delta_bg.transpose());

    for (int i = 0; i <= WINDOW_SIZE; i++) // WINDOW크기만큼 순회
        Bgs[i] += delta_bg; // 바이어스 값을 Bgs 배열에 업데이트

    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end( ); frame_i++)
    {
        frame_j = next(frame_i);
        frame_j->second.pre_integration->repropagate(Vector3d::Zero(), Bgs[0]); // bias를 통해 IMU 데이터를 재보정
    }
}


MatrixXd TangentBasis(Vector3d &g0)
{
    Vector3d b, c;
    Vector3d a = g0.normalized(); // 중력 벡터를 normalization하여 a 벡터로 함
    Vector3d tmp(0, 0, 1);
    if(a == tmp) // a 벡터와 tmp 벡터가 같다면 
        tmp << 1, 0, 0; //tmp 벡터를 (1, 0, 0)로 변경
    b = (tmp - a * (a.transpose() * tmp)).normalized(); // a벡터와 tmp를 통해 b 벡터 연산
    c = a.cross(b); // a와 b 벡터의 외적을 통해 c 벡터 연산
    MatrixXd bc(3, 2); // b와 c 벡터를 열로 갖는 3x2 크기의 행렬 bc를 생성
    bc.block<3, 1>(0, 0) = b;
    bc.block<3, 1>(0, 1) = c; 
    return bc; // bc 행렬 반환
}

void RefineGravity(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x)
{
    Vector3d g0 = g.normalized() * G.norm(); // 중력 벡터를 normalization하고 G벡터의 크기와 맞춘 후 g0로 설정
    Vector3d lx, ly;
    //VectorXd x;
    int all_frame_count = all_image_frame.size(); // 모든 이미지 수
    int n_state = all_frame_count * 3 + 2 + 1; // state의 크기 정의

    MatrixXd A{n_state, n_state}; // A행렬을 NxN으로 생성
    A.setZero();
    VectorXd b{n_state}; // B 행렬을 Nx1로 생성
    b.setZero();

    map<double, ImageFrame>::iterator frame_i; // all_image_frame에 접근하기 위한 iterator 생성
    map<double, ImageFrame>::iterator frame_j;
    for(int k = 0; k < 4; k++)
    {
        MatrixXd lxly(3, 2);
        lxly = TangentBasis(g0); // 중력 벡터를 통한 기저(basis) 생성
        int i = 0;
        for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++) // 모든 이미치 frame에 대해 순회
        {
            frame_j = next(frame_i); // frame_i가 가리키는 frame의 다음 frame을 가져옴

            MatrixXd tmp_A(6, 9); // tmp_A 및 tmp_b 행렬 생성
            tmp_A.setZero();
            VectorXd tmp_b(6);
            tmp_b.setZero();

            double dt = frame_j->second.pre_integration->sum_dt; // frame dt를 가져옴


            tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity(); // imu의 Rotation 및 Translation, TIC 등을 통해 tmp_A 행렬 구성
            tmp_A.block<3, 2>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity() * lxly; // Rotation 행렬에 중력 벡터 기저를 곱함
            tmp_A.block<3, 1>(0, 8) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;     
            tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0] - frame_i->second.R.transpose() * dt * dt / 2 * g0;

            tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
            tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
            tmp_A.block<3, 2>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity() * lxly; // Rotation 행렬에 중력 벡터 기저를 곱함
            tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v - frame_i->second.R.transpose() * dt * Matrix3d::Identity() * g0;


            Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero(); // Imu의 가중치 행렬
            //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
            //MatrixXd cov_inv = cov.inverse();
            cov_inv.setIdentity(); // 단위 행렬로 설정

            MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A; // tmp_A행렬을 전치하고 가중치와 tmp_A 행렬을 곱하여 r_A로 함
            VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b; // tmp_A행렬을 전치하고 가중치와 tmp_b 행렬을 곱하여 r_b로 함

            A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>(); // r_A를 A 행렬에 누적
            b.segment<6>(i * 3) += r_b.head<6>(); // r_b를 b 행렬에 누적

            A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
            b.tail<3>() += r_b.tail<3>();

            A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
            A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
        }
            A = A * 1000.0; // A 와 b에 1000을 곱함
            b = b * 1000.0;
            x = A.ldlt().solve(b); // A * x = b 라는 선형식을 푼다(LDLT 분해)
            VectorXd dg = x.segment<2>(n_state - 3); // x의 마지막 2개 요소를 추출하여 dg 벡터로 저장 (중력 보정 값)
            g0 = (g0 + lxly * dg).normalized() * G.norm(); // g0에 기저와 보정값을 곱하여 normalization하고 G벡터의 크기와 맞춘 후 g0로 설정
            //double s = x(n_state - 1);
    }   
    g = g0; // 연산한 g0를 인자로 들어온 g로 설정
}

bool LinearAlignment(map<double, ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x)
{
    int all_frame_count = all_image_frame.size(); //전체 이미지 프레임의 수
    int n_state = all_frame_count * 3 + 3 + 1; // state의 크기 정의

    MatrixXd A{n_state, n_state}; // A행렬을 NxN으로 생성
    A.setZero(); // 0행렬로 초기화
    VectorXd b{n_state}; // B 행렬을 Nx1로 생성
    b.setZero(); // 0행렬로 초기화

    map<double, ImageFrame>::iterator frame_i; // all_image_frame에 접근하기 위한 iterator 생성
    map<double, ImageFrame>::iterator frame_j;
    int i = 0;
    for (frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++) // 모든 이미치 frame에 대해 순회
    {
        frame_j = next(frame_i); // frame_i가 가리키는 frame의 다음 frame을 가져옴

        MatrixXd tmp_A(6, 10); // tmp_A 및 tmp_b 행렬 생성
        tmp_A.setZero();
        VectorXd tmp_b(6);
        tmp_b.setZero();

        double dt = frame_j->second.pre_integration->sum_dt; // frame dt를 가져옴

        tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity(); // imu의 Rotation 및 Translation, TIC 등을 통해 tmp_A 행렬 구성
        tmp_A.block<3, 3>(0, 6) = frame_i->second.R.transpose() * dt * dt / 2 * Matrix3d::Identity();
        tmp_A.block<3, 1>(0, 9) = frame_i->second.R.transpose() * (frame_j->second.T - frame_i->second.T) / 100.0;     
        tmp_b.block<3, 1>(0, 0) = frame_j->second.pre_integration->delta_p + frame_i->second.R.transpose() * frame_j->second.R * TIC[0] - TIC[0];
        //cout << "delta_p   " << frame_j->second.pre_integration->delta_p.transpose() << endl;
        tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity(); 
        tmp_A.block<3, 3>(3, 3) = frame_i->second.R.transpose() * frame_j->second.R;
        tmp_A.block<3, 3>(3, 6) = frame_i->second.R.transpose() * dt * Matrix3d::Identity();
        tmp_b.block<3, 1>(3, 0) = frame_j->second.pre_integration->delta_v;
        //cout << "delta_v   " << frame_j->second.pre_integration->delta_v.transpose() << endl;

        Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero(); // Imu의 가중치 행렬
        //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
        //MatrixXd cov_inv = cov.inverse();
        cov_inv.setIdentity(); // 단위 행렬로 설정

        MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A; // tmp_A행렬을 전치하고 가중치와 tmp_A 행렬을 곱하여 r_A로 함
        VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b; // tmp_A행렬을 전치하고 가중치와 tmp_b 행렬을 곱하여 r_b로 함

        A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>(); // r_A를 A 행렬에 누적
        b.segment<6>(i * 3) += r_b.head<6>(); // r_b를 b 행렬에 누적

        A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
        b.tail<4>() += r_b.tail<4>();

        A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
        A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
    }
    A = A * 1000.0; // A 와 b에 1000을 곱함
    b = b * 1000.0;
    x = A.ldlt().solve(b); // A * x = b 라는 선형식을 푼다(LDLT 분해)
    double s = x(n_state - 1) / 100.0; // x에서 Scale 값 추출
    ROS_DEBUG("estimated scale: %f", s);
    g = x.segment<3>(n_state - 4); // x에서 중력 벡터 g를 추출
    ROS_DEBUG_STREAM(" result g     " << g.norm() << " " << g.transpose());
    if(fabs(g.norm() - G.norm()) > 1.0 || s < 0) // 연산 한 값과 이미 정의된 값(9.8)의 차이가 너무 크거나 작으면 실패 또한 스케일이 0 이하면 실패
    {
        return false;
    }

    RefineGravity(all_image_frame, g, x); // 중력 벡터를 보정
    s = (x.tail<1>())(0) / 100.0; // 보정된 스케일 요소를 추출 
    (x.tail<1>())(0) = s; 
    ROS_DEBUG_STREAM(" refine     " << g.norm() << " " << g.transpose());
    if(s < 0.0 ) //스케일이 음수인 경우 실패
        return false;   
    else
        return true;
}

bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs, Vector3d &g, VectorXd &x)
{
    solveGyroscopeBias(all_image_frame, Bgs); //Imu의 gyro bias를 보정

    if(LinearAlignment(all_image_frame, g, x)) // 중력 벡터 보정, 성공적으로 보정 되었다면
        return true;
    else 
        return false;
}
