#include "initial_sfm.h"

GlobalSFM::GlobalSFM(){}

void GlobalSFM::triangulatePoint(Eigen::Matrix<double, 3, 4> &Pose0, Eigen::Matrix<double, 3, 4> &Pose1,
						Vector2d &point0, Vector2d &point1, Vector3d &point_3d)
{
	Matrix4d design_matrix = Matrix4d::Zero(); // 4x4 0행렬 초기화
	design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0); // 첫번째 행을 1번째 x 값 * 1번째 pose의 3번째 행 - 1번째 행으로
	design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1); // 두번째 행을 1번째 y 값 * 1번째 pose의 3번째 행 - 2번째 행으로
	design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0); // 세번째 행을 2번째 x 값 * 2번째 pose의 3번째 행 - 1번째 행으로
	design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1); // 네번째 행을 2번째 y 값 * 1번째 pose의 3번째 행 - 2번째 행으로
	Vector4d triangulated_point;
	triangulated_point =
		      design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>(); // design_matrix를 Singular Value Decomposition(특이값 분해)하여 triangulated_point 계산
	point_3d(0) = triangulated_point(0) / triangulated_point(3); // triangulated_point의 마지막 element를 통해 비례적으로 x, y, z값을 추정
	point_3d(1) = triangulated_point(1) / triangulated_point(3);
	point_3d(2) = triangulated_point(2) / triangulated_point(3);
}


bool GlobalSFM::solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i,
								vector<SFMFeature> &sfm_f)
{
	vector<cv::Point2f> pts_2_vector; // 2D point 배열
	vector<cv::Point3f> pts_3_vector; // 3D point 배열
	for (int j = 0; j < feature_num; j++) // 특징점 개수 만큼 반복
	{
		if (sfm_f[j].state != true) // j번째 SFMFeature 객체의 state가 true가 아닌 경우 (추적이 불가능하다면)
			continue;
		Vector2d point2d; 
		for (int k = 0; k < (int)sfm_f[j].observation.size(); k++) // SFMFeature의 observation 배열의 크기만큼 반복
		{
			if (sfm_f[j].observation[k].first == i) // observation 배열의 feature index가 i(PnP를 진행할 frame 번호)와 같은 경우
			{
				Vector2d img_pts = sfm_f[j].observation[k].second; // observation 배열에서 특징점 좌표 반환
				cv::Point2f pts_2(img_pts(0), img_pts(1)); 
				pts_2_vector.push_back(pts_2); // 2D Point 배열에 삽입
				cv::Point3f pts_3(sfm_f[j].position[0], sfm_f[j].position[1], sfm_f[j].position[2]); 
				pts_3_vector.push_back(pts_3); // 3D Point 배열에 삽입
				break;
			}
		}
	}
	if (int(pts_2_vector.size()) < 15) // 2D Point가 15개 미만인 경우
	{
		printf("unstable features tracking, please slowly move you device!\n");
		if (int(pts_2_vector.size()) < 10) // 2D Point가 10개 미만인 경우
			return false; // 실패
	}
	cv::Mat r, rvec, t, D, tmp_r;
	cv::eigen2cv(R_initial, tmp_r); // 카메라 Rotation 행렬을 opencv의 Mat 객체로 변환
	cv::Rodrigues(tmp_r, rvec); // PnP를 위해 행렬을 Rodrigues 변환
	cv::eigen2cv(P_initial, t); // 카메라 Translation 행렬을 opencv의 Mat 객체로 변환
	cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1); // K 행렬 초기화
	bool pnp_succ; 
	pnp_succ = cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1); //solvePnP를 통해 PnP 진행
	if(!pnp_succ) // PnP에 실패한 경우
	{
		return false; // 실패
	}
	cv::Rodrigues(rvec, r); // Rodrigues 변환된 행렬을 원래 형태인 행렬로 변환
	//cout << "r " << endl << r << endl;
	MatrixXd R_pnp; 
	cv::cv2eigen(r, R_pnp); // Roatation 행렬을 Matrix3d 객체로 변환
	MatrixXd T_pnp;
	cv::cv2eigen(t, T_pnp); // Translation 행렬을 Matrix3d 객체로 변환
	R_initial = R_pnp; // Rotation 및 Translation 행렬 복사
	P_initial = T_pnp;
	return true; // 성공

}

void GlobalSFM::triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0, 
									 int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
									 vector<SFMFeature> &sfm_f)
{
	assert(frame0 != frame1); //frame0의 index와 frame1의 index가 같으면 종료
	for (int j = 0; j < feature_num; j++) // 특징점 개수만큼 반복
	{
		if (sfm_f[j].state == true) // SFMFeature 객체의 state가 true인 경우
			continue; // 무시
		bool has_0 = false, has_1 = false;
		Vector2d point0;
		Vector2d point1;
		for (int k = 0; k < (int)sfm_f[j].observation.size(); k++) // SFMFeature 객체의 observation 배열 크기만큼 반복
		{
			if (sfm_f[j].observation[k].first == frame0) // observation 배열 속 feature의 index가 frame0 값과 같다면
			{
				point0 = sfm_f[j].observation[k].second; // feature point를 가져온다.
				has_0 = true; // feature가 있다고 frag 설정
			}
			if (sfm_f[j].observation[k].first == frame1) // observation 배열 속 feature의 index가 frame1 값과 같다면
			{
				point1 = sfm_f[j].observation[k].second; // feature point를 가져온다.
				has_1 = true; // feature가 있다고 frag 설정
			}
		}
		if (has_0 && has_1) // 2개의 feature point가 존재한다면
		{
			Vector3d point_3d;
			triangulatePoint(Pose0, Pose1, point0, point1, point_3d); // point를 통해 triangulation 진행
			sfm_f[j].state = true; // state를 true로 변환
			sfm_f[j].position[0] = point_3d(0); // SFMFeature 객체의 point를 triangulation로 추정한 point로 변경
			sfm_f[j].position[1] = point_3d(1);
			sfm_f[j].position[2] = point_3d(2);
			//cout << "trangulated : " << frame1 << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
		}							  
	}
}
// construct(frame_count + 1, Q, T, l,
// relative_R, relative_T,
// sfm_f, sfm_tracked_points
// 	 q w_R_cam t w_R_cam
//  c_rotation cam_R_w 
//  c_translation cam_R_w
// relative_q[i][j]  j_q_i
// relative_t[i][j]  j_t_ji  (j < i)
bool GlobalSFM::construct(int frame_num, Quaterniond* q, Vector3d* T, int l,
			  const Matrix3d relative_R, const Vector3d relative_T,
			  vector<SFMFeature> &sfm_f, map<int, Vector3d> &sfm_tracked_points)
{
	feature_num = sfm_f.size(); // SFMFeature 객체를 담고있는 배열의 크기
	//cout << "set 0 and " << l << " as known " << endl;
	// have relative_r relative_t
	// intial two view
	q[l].w() = 1; // 상대 frame 인덱스의 카메라 회전 행렬의 w(스칼라) 값 초기화
	q[l].x() = 0; // 상대 frame 인덱스의 카메라 회전 행렬의 x, y,z 값 초기화
	q[l].y() = 0;
	q[l].z() = 0;
	T[l].setZero(); // 상대 frame 인덱스의 카메라 위치 행렬을 0행렬로 초기화 
	q[frame_num - 1] = q[l] * Quaterniond(relative_R); // 이전 프레임의 카메라 회전 행렬 = 상대 frame의 카메라 행렬 * 상대 회전 행렬 
	T[frame_num - 1] = relative_T; // 이전 프레임의 카메라 위치 행렬 = 상대 frame의 카메라 행렬 * 상대 회전 행렬 
	//cout << "init q_l " << q[l].w() << " " << q[l].vec().transpose() << endl;
	//cout << "init t_l " << T[l].transpose() << endl;

	//rotate to cam frame
	Matrix3d c_Rotation[frame_num]; 
	Vector3d c_Translation[frame_num];
	Quaterniond c_Quat[frame_num];
	double c_rotation[frame_num][4];
	double c_translation[frame_num][3];
	Eigen::Matrix<double, 3, 4> Pose[frame_num];

	c_Quat[l] = q[l].inverse(); // 카메라 회전 행렬의 역수를 저장 
	c_Rotation[l] = c_Quat[l].toRotationMatrix(); // 쿼터니언인 카메라 회전 행렬을 Matrix3d 객체로 변환
	c_Translation[l] = -1 * (c_Rotation[l] * T[l]); // 회전 행렬과 위치 행렬을 곱해 Translation 행렬 생성
	Pose[l].block<3, 3>(0, 0) = c_Rotation[l]; // Rotation 행렬과 Translation 행렬을 통해 l 번째 카메라 pose 구성
	Pose[l].block<3, 1>(0, 3) = c_Translation[l];

	c_Quat[frame_num - 1] = q[frame_num - 1].inverse(); // 이전 프레임에 대해 카메라 pose 구성
	c_Rotation[frame_num - 1] = c_Quat[frame_num - 1].toRotationMatrix();
	c_Translation[frame_num - 1] = -1 * (c_Rotation[frame_num - 1] * T[frame_num - 1]);
	Pose[frame_num - 1].block<3, 3>(0, 0) = c_Rotation[frame_num - 1];
	Pose[frame_num - 1].block<3, 1>(0, 3) = c_Translation[frame_num - 1];


	//1: trangulate between l ----- frame_num - 1
	//2: solve pnp l + 1; trangulate l + 1 ------- frame_num - 1; 
	for (int i = l; i < frame_num - 1 ; i++)
	{
		// solve pnp
		if (i > l) //l번째 프레임 이후인 경우
		{
			Matrix3d R_initial = c_Rotation[i - 1];
			Vector3d P_initial = c_Translation[i - 1];
			if(!solveFrameByPnP(R_initial, P_initial, i, sfm_f)) // PnP를 통해 Rotation 행렬과 Translation 행렬을 추정
				return false; // PnP 실패 시 false 반환
			c_Rotation[i] = R_initial; // 카메라 행렬을 추정한 값으로 변경
			c_Translation[i] = P_initial; 
			c_Quat[i] = c_Rotation[i];
			Pose[i].block<3, 3>(0, 0) = c_Rotation[i]; // pose 재 구성 
			Pose[i].block<3, 1>(0, 3) = c_Translation[i];
		}

		// triangulate point based on the solve pnp result
		triangulateTwoFrames(i, Pose[i], frame_num - 1, Pose[frame_num - 1], sfm_f); // PnP를 통해 구한 pose를 이용하여 trianglate 계산
	}
	//3: triangulate l-----l+1 l+2 ... frame_num -2
	for (int i = l + 1; i < frame_num - 1; i++) // l + 1번째 frame과 frame_num -2 사이 point를 triangulate
		triangulateTwoFrames(l, Pose[l], i, Pose[i], sfm_f);
	//4: solve pnp l-1; triangulate l-1 ----- l
	//             l-2              l-2 ----- l
	for (int i = l - 1; i >= 0; i--) // l - 1 부터 0까지 반복
	{
		//solve pnp
		Matrix3d R_initial = c_Rotation[i + 1]; // 카메라 Rotation 행렬
		Vector3d P_initial = c_Translation[i + 1]; // 카메라 Translation 행렬
		if(!solveFrameByPnP(R_initial, P_initial, i, sfm_f)) // PnP를 통해 Rotation 행렬과 Translation 행렬을 추정
			return false;
		c_Rotation[i] = R_initial; // 카메라 행렬을 추정한 값으로 변경
		c_Translation[i] = P_initial;
		c_Quat[i] = c_Rotation[i];
		Pose[i].block<3, 3>(0, 0) = c_Rotation[i]; // pose 재 구성 
		Pose[i].block<3, 1>(0, 3) = c_Translation[i];
		//triangulate
		triangulateTwoFrames(i, Pose[i], l, Pose[l], sfm_f); // PnP를 통해 구한 pose를 이용하여 trianglate 계산
	}
	//5: triangulate all other points
	for (int j = 0; j < feature_num; j++) // 모든 feature에 대해 반복
	{
		if (sfm_f[j].state == true) // SFMFeature 객체의 state가 true인 경우 무시
			continue;
		if ((int)sfm_f[j].observation.size() >= 2) // observation 배열의 크기가 2 이상인 경우
		{
			Vector2d point0, point1;
			int frame_0 = sfm_f[j].observation[0].first; // observation 배열에서 0번째 요소 index를 가져옴
			point0 = sfm_f[j].observation[0].second; // observation 배열에서 0번째 요소의 point를 가져옴
			int frame_1 = sfm_f[j].observation.back().first; // observation 배열에서 마지막 요소 index를 가져옴
			point1 = sfm_f[j].observation.back().second; // observation 배열에서 마지막 요소의 point를 가져옴
			Vector3d point_3d;
			triangulatePoint(Pose[frame_0], Pose[frame_1], point0, point1, point_3d); // triangulate 진행
			sfm_f[j].state = true; // 상태를 true로 변경
			sfm_f[j].position[0] = point_3d(0); // position을 추정한 값으로 변경
			sfm_f[j].position[1] = point_3d(1);
			sfm_f[j].position[2] = point_3d(2);
			//cout << "trangulated : " << frame_0 << " " << frame_1 << "  3d point : "  << j << "  " << point_3d.transpose() << endl;
		}		
	}

/*
	for (int i = 0; i < frame_num; i++)
	{
		q[i] = c_Rotation[i].transpose(); 
		cout << "solvePnP  q" << " i " << i <<"  " <<q[i].w() << "  " << q[i].vec().transpose() << endl;
	}
	for (int i = 0; i < frame_num; i++)
	{
		Vector3d t_tmp;
		t_tmp = -1 * (q[i] * c_Translation[i]);
		cout << "solvePnP  t" << " i " << i <<"  " << t_tmp.x() <<"  "<< t_tmp.y() <<"  "<< t_tmp.z() << endl;
	}
*/
	//full BA
	ceres::Problem problem; // Ceres Solver의 최적화 문제를 정의하는 객체 생성 (Bundle Adjustment를 Ceres Solver라는 라이브러리를 사용해서 해결)
	ceres::LocalParameterization* local_parameterization = new ceres::QuaternionParameterization(); //쿼터니언 변수가 회전 행렬의 성질을 유지하도록 제한하는 객체 생성
	//cout << " begin full BA " << endl;
	for (int i = 0; i < frame_num; i++) // 모든 frame에 대해 순회
	{
		//double array for ceres
		c_translation[i][0] = c_Translation[i].x(); // c_Translation의 x, y, z 값을 Ceres Solver가 사용하는 형태로 변환하기 위해 c_translation에 저장
		c_translation[i][1] = c_Translation[i].y();
		c_translation[i][2] = c_Translation[i].z();
		c_rotation[i][0] = c_Quat[i].w(); // Ceres Solver가 사용하는 배열 형태로 c_Quat를 저장
		c_rotation[i][1] = c_Quat[i].x();
		c_rotation[i][2] = c_Quat[i].y();
		c_rotation[i][3] = c_Quat[i].z();
		problem.AddParameterBlock(c_rotation[i], 4, local_parameterization); //회전과 위치 배열을 최적화 대상으로 추가
		problem.AddParameterBlock(c_translation[i], 3);
		if (i == l) // i가 l과 같을 경우
		{
			problem.SetParameterBlockConstant(c_rotation[i]); // l 프레임을 기준 프레임으로 사용, 기준 프레임의 회전을 바꾸지 않도록 고정
		}
		if (i == l || i == frame_num - 1)
		{
			problem.SetParameterBlockConstant(c_translation[i]); // l과 마지막 프레임의 위치 변수를 상수로 설정, 기준 프레임과 마지막 프레임의 위치를 바꾸지 않도록 고정
		}
	}

	for (int i = 0; i < feature_num; i++) // 모든 feature point에 대해 반복
	{
		if (sfm_f[i].state != true) // SFMFeature 객체의 state가 true인 경우 무시
			continue;
		for (int j = 0; j < int(sfm_f[i].observation.size()); j++) // observation 배열의 크기만큼 반복
		{
			int l = sfm_f[i].observation[j].first; // frame index를 가져온다.
			ceres::CostFunction* cost_function = ReprojectionError3D::Create(
												sfm_f[i].observation[j].second.x(),
												sfm_f[i].observation[j].second.y()); //featue의 x, y 좌표를 통해 재투영 오차를 계산하는 CostFunction을 생성

    		problem.AddResidualBlock(cost_function, NULL, c_rotation[l], c_translation[l], 
    								sfm_f[i].position);	 // 재투영 오차를 최소화하는 Residual Block을 추가, 카메라 rotation, 카메라 translation, position을 최적화 대상으로
		}
		 
	}
	ceres::Solver::Options options; // Ceres Solver의 옵션을 설정하는 구조체 생성
	options.linear_solver_type = ceres::DENSE_SCHUR; // 최적화에 사용할 liner_solver_type 선언, DENSE_SCHUR 타입을 사용
	//options.minimizer_progress_to_stdout = true;
	options.max_solver_time_in_seconds = 0.2; // 최적화를 수행할 최대 시간을 설정
	ceres::Solver::Summary summary; // 최적화 결과를 저장할 구조체 변수
	ceres::Solve(options, &problem, &summary); // Ceres Solver를 사용하여 최적화를 수행
	//std::cout << summary.BriefReport() << "\n";
	// 최적화 종료 타입이 ceres::CONVERGENCE(최적화 성공)이거나 최적화 연산 시 오차가 5e-03보다 작은 경우
	if (summary.termination_type == ceres::CONVERGENCE || summary.final_cost < 5e-03)
	{
		//cout << "vision only BA converge" << endl;
	}
	else // 최적화 실패
	{
		//cout << "vision only BA not converge " << endl;
		return false;
	}
	for (int i = 0; i < frame_num; i++) // 모든 프레임에 대해 반복
	{
		q[i].w() = c_rotation[i][0]; // Ceres Solver에서 최적화한 rotation 행렬을 q 행렬에 복사
		q[i].x() = c_rotation[i][1]; 
		q[i].y() = c_rotation[i][2]; 
		q[i].z() = c_rotation[i][3]; 
		q[i] = q[i].inverse(); //역수를 취하여 원래 형태로 변환 (함수 초기에 q 배열에 대해 inverse 연산을 하여 c_Quat에 저장하였기에)
		//cout << "final  q" << " i " << i <<"  " <<q[i].w() << "  " << q[i].vec().transpose() << endl;
	}
	for (int i = 0; i < frame_num; i++) // 모든 프레임에 대해 반복
	{

		T[i] = -1 * (q[i] * Vector3d(c_translation[i][0], c_translation[i][1], c_translation[i][2])); // 최적화된 translation 행렬에 대해 q와의 곱셈과 음수를 취해 원래의 위치를 계산 
		//cout << "final  t" << " i " << i <<"  " << T[i](0) <<"  "<< T[i](1) <<"  "<< T[i](2) << endl;
	}
	for (int i = 0; i < (int)sfm_f.size(); i++)
	{
		if(sfm_f[i].state) // state가 true인 경우
			sfm_tracked_points[sfm_f[i].id] = Vector3d(sfm_f[i].position[0], sfm_f[i].position[1], sfm_f[i].position[2]); //추정한 점의 좌표를 저장하는 map 객체에 추정된 점의 좌표 저장
	}
	return true; // true 반환

}

