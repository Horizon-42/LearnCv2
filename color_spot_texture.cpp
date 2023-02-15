#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
    // 加载图像
    Mat img = imread("example.jpg");

    // 转换为灰度图像
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    // 计算灰度共生矩阵
    Mat glcm;
    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8, 8));
    clahe->apply(gray, gray);
    calcHist(&gray, 1, 0, Mat(), glcm, 1, &256, 1, true, false);
    glcm /= glcm.at<float>(0, 0); // 归一化

    // 使用kmeans聚类
    Mat samples(glcm.rows, 1, CV_32FC1);
    for (int i = 0; i < glcm.rows; i++) {
        samples.at<float>(i, 0) = glcm.at<float>(i, 0);
    }
    int K = 2;
    Mat labels, centers;
    kmeans(samples, K, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0), 3, KMEANS_PP_CENTERS, centers);

    // 根据标签图像分割图像并绘制色斑轮廓
    for (int i = 0; i < 2; i++) {
        Mat mask = labels == i;
        vector<vector<Point>> contours;
        findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        // 绘制色斑轮廓
        for (auto& contour : contours) {
            double area = contourArea(contour);
            if (area < 50) { // 忽略面积较小的轮廓
                continue;
            }
            Rect rect = boundingRect(contour);
            rectangle(img, rect, Scalar(0, 0, 255), 2);
        }
    }

    // 显示结果
    imshow("result", img);
    waitKey(0);
    destroyAllWindows();

    return 0;
}
