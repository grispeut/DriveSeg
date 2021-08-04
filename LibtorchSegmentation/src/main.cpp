#include "log.h"
#include "params.h"
#include "DriveSegmentation.h"
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netdb.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <iostream>
#include <errno.h>

//#include <sl/Camera.hpp>

using namespace std;
//using namespace sl;
//cv::Mat slMat2cvMat(Mat& input);

void error(const char *msg)
{
    perror(msg);
    exit(0);
}

int main(int argc, char* argv[])
{
    int tcp_mode, debug_mode=1;
//    sscanf(argv[1], "%d", &tcp_mode);
//    sscanf(argv[2], "%d", &debug_mode);

//    Camera zed;
//    InitParameters init_parameters;
//    init_parameters.camera_resolution = RESOLUTION::HD720; // Use HD1080 video mode
//    init_parameters.camera_fps = 10;
//    init_parameters.depth_mode = DEPTH_MODE::PERFORMANCE; // Use PERFORMANCE depth mode
//    init_parameters.coordinate_units = UNIT::MILLIMETER; // Use millimeter units (for depth measurements)
//    auto returned_state = zed.open(init_parameters);
//    if (returned_state != ERROR_CODE::SUCCESS) {
//        cout << "Error " << returned_state << ", exit program." << endl;
//        //return EXIT_FAILURE;
//    }
//    RuntimeParameters runtime_parameters;
//    runtime_parameters.sensing_mode = SENSING_MODE::STANDARD; // Use STANDARD sensing mode
//    sl::Mat image, depth, point_cloud;
//
    cv::Mat img_zed2 = cv::Mat::zeros(720, 1280, CV_8UC3);
    cv::Mat img_zed2_depth = cv::Mat::zeros(720, 1280, CV_16UC1);
//
//    int sockfd_send, portno_send;
//    struct sockaddr_in serv_addr_send;
//    sockfd_send = socket(AF_INET, SOCK_STREAM, 0);
//    bzero((char *) &serv_addr_send, sizeof(serv_addr_send));
//    serv_addr_send.sin_family = AF_INET;
//    if (tcp_mode==-1)serv_addr_send.sin_addr.s_addr = inet_addr("127.0.0.1");
//    else serv_addr_send.sin_addr.s_addr = inet_addr("192.168.1.50");
//    serv_addr_send.sin_port = htons(50000);
//    while(1){
//        if (connect(sockfd_send,(struct sockaddr *)&serv_addr_send,sizeof(serv_addr_send))>=0)
//            break;
//    }

    TModelParam modelParam;
    modelParam.modelPath = "../../weights/model_half.pt";
    modelParam.meanValue = {0.279, 0.293, 0.290};
    modelParam.stdValue = {0.197, 0.198, 0.201};
    modelParam.deviceType = torch::DeviceType::CUDA;
    modelParam.gpuId = 0;
    DriveSegmentation  ptDriveSeg;
    bool is_load = ptDriveSeg.InitModel(&modelParam);
    if (!is_load)
    {
        LOG_ERROR("The model loading failed!");
    }

    while(1){
        img_zed2 = cv::imread("../samples/zed2.jpg", -1);
	    img_zed2_depth = cv::imread("../depth.png", -1);
//        if (zed.grab(runtime_parameters) == ERROR_CODE::SUCCESS) {
//            // Retrieve left image
//            zed.retrieveImage(image, VIEW::LEFT);
//            // Retrieve depth map. Depth is aligned on the left image
//            zed.retrieveMeasure(depth, MEASURE::DEPTH);
//            // Retrieve colored point cloud. Point cloud is aligned on the left image.
//            zed.retrieveMeasure(point_cloud, MEASURE::XYZRGBA);
//
//            img_zed2= slMat2cvMat(image);
//            img_zed2_depth = slMat2cvMat(depth);
//
//            // Get and print distance value in mm at the center of the image
//            // We measure the distance camera - object using Euclidean distance
//            int x = 600;
//            int y = 300;
//            sl::float4 point_cloud_value;
//            point_cloud.getValue(x, y, &point_cloud_value);
//
//            if(std::isfinite(point_cloud_value.z)){
//                float distance = sqrt(point_cloud_value.x * point_cloud_value.x + point_cloud_value.y * point_cloud_value.y + point_cloud_value.z * point_cloud_value.z);
//                cout<<"Distance to Camera at {"<<x<<";"<<y<<"}: "<<point_cloud_value.z<<"mm,"<<distance<<"mm"<<endl;
//            }else
//                cout<<"The Distance can not be computed at {"<<x<<";"<<y<<"}"<<endl;
//	    cv::cvtColor(img_zed2, img_zed2, cv::COLOR_BGRA2BGR);
//	    //if (debug_mode==1) cv::imwrite("../depth_ori.png", img_zed2_depth);
//	    img_zed2_depth.convertTo(img_zed2_depth, CV_16UC1);
//	    if (debug_mode==1) cv::imwrite("../depth.png", img_zed2_depth);
//	    if (debug_mode==1) cv::imwrite("../zed2.png", img_zed2);
//        }
//	else{
//	    img_zed2 = cv::imread("../zed2.png", -1);
//	    img_zed2_depth = cv::imread("../depth.png", -1);
//	}


        cv::Mat image;
        cv::resize(img_zed2, image, cv::Size(512, 512));
        if (image.empty())
        {
            LOG_ERROR("The image read failed!");
        }
        struct timeval tp;
        struct timeval tp1;
        int start;
        int end;
        torch::Tensor output;
        gettimeofday(&tp, NULL);
        start = tp.tv_sec*1000 + tp.tv_usec/1000;
        int iter_num = 200;
        for (int i = 0; i < iter_num; ++i)
        {
            output = ptDriveSeg.Forward(image);
        }
        gettimeofday(&tp1, NULL);
        end = tp1.tv_sec * 1000 + tp1.tv_usec/1000;
        std::cout << (end -start) / iter_num << std::endl;
        if (output.numel() < image.rows*image.cols)
        {
            LOG_ERROR("The output is invalid!");
        }
        cv::Mat mask = cv::Mat(image.rows, image.cols, CV_8UC1);
        memcpy(mask.data, output.data_ptr(), output.numel()*sizeof(torch::kU8));
        cv::resize(mask, mask, cv::Size(1280, 720));
        if (debug_mode==1) cv::imwrite("../mask.png", mask);
        break;

//        cv::Mat mask_scp = cv::Mat(image.rows, image.cols, CV_8UC1);
//        //cv::cvtColor(mask,mask_rgb,COLOR_GRAY2RGB);
//        mask_scp = (mask.reshape(0,1));
//        ssize_t imgSize_mask = mask_scp.total()*mask_scp.elemSize();
//        if(! mask_scp.data ){cout <<  "Could not open or find the image" << endl ;}
//        send(sockfd_send, mask_scp.data, imgSize_mask, 0);
//        img_zed2_depth = (img_zed2_depth.reshape(0,1));
//        ssize_t imgSize_depth = img_zed2_depth.total()*img_zed2_depth.elemSize();
//        // Send data here
//        send(sockfd_send, img_zed2_depth.data, imgSize_depth, 0);
    }

    return 0;
}


//int getOCVtype(sl::MAT_TYPE type) {
//    int cv_type = -1;
//    switch (type) {
//        case MAT_TYPE::F32_C1: cv_type = CV_32FC1; break;
//        case MAT_TYPE::F32_C2: cv_type = CV_32FC2; break;
//        case MAT_TYPE::F32_C3: cv_type = CV_32FC3; break;
//        case MAT_TYPE::F32_C4: cv_type = CV_32FC4; break;
//        case MAT_TYPE::U8_C1: cv_type = CV_8UC1; break;
//        case MAT_TYPE::U8_C2: cv_type = CV_8UC2; break;
//        case MAT_TYPE::U8_C3: cv_type = CV_8UC3; break;
//        case MAT_TYPE::U8_C4: cv_type = CV_8UC4; break;
//        default: break;
//    }
//    return cv_type;
//}


//cv::Mat slMat2cvMat(Mat& input) {
//    // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
//    // cv::Mat and sl::Mat will share a single memory structure
//    return cv::Mat(input.getHeight(), input.getWidth(), getOCVtype(input.getDataType()), input.getPtr<sl::uchar1>(MEM::CPU), input.getStepBytes(sl::MEM::CPU));
//}
