#include "feature_tracker.h"

int FeatureTracker::n_id = 0;

bool inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1; // 이미지 경계에서 허용되는 여백의 크기 설정
    int img_x = cvRound(pt.x); // 인자로 들어온 좌표를 정수로 반올림
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE; // point가 이미지 경계 내에 있는지 확인
}

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++) // 인자로 들어온 배열을 순회
        if (status[i]) // status가 true인 경우
            v[j++] = v[i]; // 현재 요소를 v[j]위치에 복사 (즉, true인 값들만 배열에 담게 된다)
    v.resize(j); // 배열 크기 변경
}

void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++) // 오버라이딩 되어서 같은 함수
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}


FeatureTracker::FeatureTracker()
{
}

void FeatureTracker::setMask()
{
    if(FISHEYE) // FishEye 카메라인 경우
        mask = fisheye_mask.clone(); // mask를 fisheye 마스크로 함
    else
        mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255)); // frame의 row와 col에 맞게 mask를 생성
    

    // prefer to keep features that are tracked for long time
    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < forw_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(forw_pts[i], ids[i]))); // 해당 특징점이 추적된 횟수, 픽셀 좌표, 추적된 특징점의 인덱스를 저장

    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first; // 추적 횟수를 기반으로 내림 차순으로 배열을 정렬
         });

    forw_pts.clear(); // 배열 초기화
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id) // 정렬한 배열을 순회
    {
        if (mask.at<uchar>(it.second.first) == 255) // feature point를 mask와 대응 시켰을 때 255인 경우 
        {
            forw_pts.push_back(it.second.first); // point를 배열에 삽입
            ids.push_back(it.second.second); // index를 배열에 삽입
            track_cnt.push_back(it.first); // 추적 횟수를 배열에 삽입
            cv::circle(mask, it.second.first, MIN_DIST, 0, -1); // 이미 선택된 point는 mask에서 해당 영역을 0으로 설정하여 중복 선택을 방지
        }
    }
}

void FeatureTracker::addPoints()
{
    for (auto &p : n_pts) // 새롭게 구한 feature 배열을 순회
    {
        forw_pts.push_back(p); // feature point 배열에 feature를 삽입
        ids.push_back(-1); // feature index를 -1로 초기화
        track_cnt.push_back(1); // 추적 count를 1로 초기화
    }
}

void FeatureTracker::readImage(const cv::Mat &_img, double _cur_time)
{
    cv::Mat img; // frame을 담을 Mat 객체 생성
    TicToc t_r; // 현재 살행 시간 저장
    cur_time = _cur_time; // 인자로 들어온 time stamp 저장

    if (EQUALIZE) // config 파일에서 EQUAIZE를 1로 한 경우
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8)); // Equalization을 위한 CLAHE(Contrast Limited Adaptive Histogram Equalization)객체 생성
        TicToc t_c; // 현재 시간 저장
        clahe->apply(_img, img); // img에 Equalization된 frame 저장
        ROS_DEBUG("CLAHE costs: %fms", t_c.toc());
    }
    else
        img = _img; // 인자로 들어온 frame 저장

    if (forw_img.empty()) // forw_img가 비어 있다면
    {
        prev_img = cur_img = forw_img = img; // 이전 이미지, 현재 이미지, forward img를 현재 이미지로 저장 (처음 이미지를 설정)
    }
    else
    {
        forw_img = img; // forward img를 현재 이미지로 설정
    }

    forw_pts.clear(); 

    if (cur_pts.size() > 0) // 현재 특징점 위치 배열에 값이 있는 경우
    {
        TicToc t_o; // 실행시간 저장
        vector<uchar> status;
        vector<float> err;
        //이전 frame, forward frame, 현재 frame에서 특징점 위치, forward 프레임에서 특징점 위치, 특징점 추적에 성공한 경우 1이 설정되는 배열, 추적 오차 행렬, 윈도우 크기, 피라미드 크기를
        // 통해 옵티컬 플로우를 진행 (이전 이미지와 현재 이미지 간 특징점들의 움직임을 추척)
        cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, cv::Size(21, 21), 3); 

        for (int i = 0; i < int(forw_pts.size()); i++) // forward frame 특징점 배열 순회
            if (status[i] && !inBorder(forw_pts[i])) // 특징점 추적에 성공하였고, point가 이미지 경계 내에 있다면
                status[i] = 0; // 추적 상태를 0으로 변경
        reduceVector(prev_pts, status); // status가 true인 점들만 남기고 배열 크기 변경
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(ids, status);
        reduceVector(cur_un_pts, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("temporal optical flow costs: %fms", t_o.toc());
    }

    for (auto &n : track_cnt) // tracking 배열을 순회하며 count값 증가
        n++;

    if (PUB_THIS_FRAME) // Publish할 Frame인 경우
    {
        rejectWithF(); // Fundamental Matrix를 사용하여 기존 특징점들을 필터링
        ROS_DEBUG("set mask begins");
        TicToc t_m; // 현재 실행 시간 저장
        setMask(); // 추적에 사용할 적절한 feature를 선택
        ROS_DEBUG("set mask costs %fms", t_m.toc());

        ROS_DEBUG("detect feature begins");
        TicToc t_t; 
        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size()); // MAX_CNT(150)와 feature point의 개수의 차이를 저장
        if (n_max_cnt > 0) // feature point의 수가 최대 값이 도달하지 않았다면
        {
            if(mask.empty())
                cout << "mask is empty " << endl;
            if (mask.type() != CV_8UC1)
                cout << "mask type wrong " << endl;
            if (mask.size() != forw_img.size())
                cout << "wrong size " << endl;
            cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), 0.01, MIN_DIST, mask); // 현재 이미지에서 새로운 feature를 추출, 감지되는 특징점의 개수는 MAX_CNT - forw_pts.size()로 제한
        }
        else
            n_pts.clear(); // 최대치인 경우 이전에 구한 새로운 feature를 버린다.
        ROS_DEBUG("detect feature costs: %fms", t_t.toc());

        ROS_DEBUG("add feature begins");
        TicToc t_a;
        addPoints(); // 새로운 frame에서 새로운 feature들이 이전 feature들과 함께 forw_pts, ids, track_cnt에 추가되어 추적 상태 및 정보가 갱신
        ROS_DEBUG("selectFeature costs: %fms", t_a.toc());
    }
    prev_img = cur_img; // 이전 frame을 현재 frame으로
    prev_pts = cur_pts; // 이전 point 배열을 현재 point 배열로
    prev_un_pts = cur_un_pts; // 이전 undistorted 배열을 현재 배열로
    cur_img = forw_img; // 현재 frame을 forward frame으로
    cur_pts = forw_pts; // 현재 point 배열을 forward point 배열로
    undistortedPoints();
    prev_time = cur_time;
}

void FeatureTracker::rejectWithF()
{
    if (forw_pts.size() >= 8) // forward feature point가 8개 이상 존재하는 경우
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f; // 현재 실생 시간 저장
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size()); // feature point를 담은 배열의 크기만큼 배열 생성
        for (unsigned int i = 0; i < cur_pts.size(); i++) // 현재 feature point 배열 크기만큼 반복
        {
            Eigen::Vector3d tmp_p;
            m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p); // camera calibration객체를 통해 현재 feature point를 Reprojection
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0; // 평면으로 재 보정한 x, y 좌표를 통해 undistorted point 생성
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_forw_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;
        cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status); // undistorted인 현재 feature들과 forward feature간에 RANSAC을 통한 Funamental Matrix 계산
        int size_a = cur_pts.size();
        reduceVector(prev_pts, status);  // status가 true인 점 (Fundamental Matrix를 만족하는 feature) 들만 남기고 배열 크기 변경
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}

bool FeatureTracker::updateID(unsigned int i)
{
    if (i < ids.size()) 
    {
        if (ids[i] == -1)
            ids[i] = n_id++;
        return true;
    }
    else
        return false;
}

void FeatureTracker::readIntrinsicParameter(const string &calib_file)
{
    ROS_INFO("reading paramerter of camera %s", calib_file.c_str()); // config파일 이름을 출력
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file); //CameraFactory 객체를 가져와 config파일에 있는 카메라 model_type(주로 pinhole)에 맞는 플레그를 설정
}

void FeatureTracker::showUndistortion(const string &name)
{
    cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < COL; i++)
        for (int j = 0; j < ROW; j++)
        {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.push_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
            //printf("%f,%f->%f,%f,%f\n)\n", a.x(), a.y(), b.x(), b.y(), b.z());
        }
    for (int i = 0; i < int(undistortedp.size()); i++)
    {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
        pp.at<float>(2, 0) = 1.0;
        //cout << trackerData[0].K << endl;
        //printf("%lf %lf\n", p.at<float>(1, 0), p.at<float>(0, 0));
        //printf("%lf %lf\n", pp.at<float>(1, 0), pp.at<float>(0, 0));
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 && pp.at<float>(0, 0) + 300 < COL + 600)
        {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) = cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
        else
        {
            //ROS_ERROR("(%f %f) -> (%f %f)", distortedp[i].y, distortedp[i].x, pp.at<float>(1, 0), pp.at<float>(0, 0));
        }
    }
    cv::imshow(name, undistortedImg);
    cv::waitKey(0);
}

void FeatureTracker::undistortedPoints()
{
    cur_un_pts.clear(); // undistorted point배열을 초기화
    cur_un_pts_map.clear();
    //cv::undistortPoints(cur_pts, un_pts, K, cv::Mat());
    for (unsigned int i = 0; i < cur_pts.size(); i++) // 현재 feature point 배열을 순회
    {
        Eigen::Vector2d a(cur_pts[i].x, cur_pts[i].y); // feature의 point를 저장
        Eigen::Vector3d b;
        m_camera->liftProjective(a, b); // Reprojection을 진행
        cur_un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z())); // 평면좌표로 다시 변환하여 좌표 보정
        cur_un_pts_map.insert(make_pair(ids[i], cv::Point2f(b.x() / b.z(), b.y() / b.z())));
        //printf("cur pts id %d %f %f", ids[i], cur_un_pts[i].x, cur_un_pts[i].y);
    }
    // caculate points velocity
    if (!prev_un_pts_map.empty()) // 이전 frame에서의 feature point를 구하였다면
    {
        double dt = cur_time - prev_time; // dt를 현재 frame과 이전 frame의 시간 차이로 설정
        pts_velocity.clear();
        for (unsigned int i = 0; i < cur_un_pts.size(); i++)  // 특징점 point 배열 순회
        {
            if (ids[i] != -1) // index가 -1이 아닌 경우
            {
                std::map<int, cv::Point2f>::iterator it;
                it = prev_un_pts_map.find(ids[i]);
                if (it != prev_un_pts_map.end())  // 이전 feature point map에서 ids[i]에 해당하는 point가 존재한다면
                {
                    double v_x = (cur_un_pts[i].x - it->second.x) / dt; // feature point와 시간차이를 통해 속도 계산
                    double v_y = (cur_un_pts[i].y - it->second.y) / dt;
                    pts_velocity.push_back(cv::Point2f(v_x, v_y)); // 속도 저장
                }
                else
                    pts_velocity.push_back(cv::Point2f(0, 0)); // 없으면 속도 0
            }
            else
            {
                pts_velocity.push_back(cv::Point2f(0, 0)); // index가 -1인 경우 속도 0
            }
        }
    }
    else
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0)); // // 이전 frame에서의 feature point를 구하지 못하면 속도 0
        }
    }
    prev_un_pts_map = cur_un_pts_map; // 이전 feature point를 현재 feature point로 설정
}
