#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void Imshow(std::string const& name, cv::Mat const& src){
    cv::namedWindow(name, cv::WINDOW_GUI_NORMAL);
    cv::imshow(name, src);
}

int main() {
    // Load image and convert to grayscale
    Mat img = imread("/home/skin/face_data/test0/Rgb_Cool.png");
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    // Apply median filter to remove noise and small details
    Mat median;
    medianBlur(gray, median, 5);

    // Apply edge detection using Sobel operator
    Mat sobelx, sobely, gradient;
    Sobel(median, sobelx, CV_64F, 1, 0, 3);
    Sobel(median, sobely, CV_64F, 0, 1, 3);
    magnitude(sobelx, sobely, gradient);
    gradient.convertTo(gradient, CV_8UC3);
    // std::cout<<gradient<<"\n";

    // Threshold gradient magnitude to obtain binary image
    double thresh = 50.0;
    Mat binary;
    cv::threshold(gradient, binary, thresh, 255, THRESH_BINARY);

    // Apply dilation and erosion to connect and shrink edges
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    Mat closed;
    morphologyEx(binary, closed, MORPH_CLOSE, kernel);

    std::cout<< (closed.type()==CV_8UC3)<<"\n";

    // Apply non-maximum suppression to detect pores
    int radius = 15;
    Mat suppressed = Mat::zeros(closed.size(), closed.type());
    for (int y = radius; y < closed.rows - radius; y++) {
        for (int x = radius; x < closed.cols - radius; x++) {
            if (closed.at<uchar>(y, x) == 255) {
                Mat patch = closed(Rect(x - radius, y - radius, 2 * radius + 1, 2 * radius + 1));
                if (countNonZero(patch == 255) >= 1) {
                    suppressed.at<uchar>(y, x) = 255;
                }
            }
        }
    }

    // Draw circles on original image to mark pore locations
    vector<Point> pores;
    findNonZero(suppressed, pores);
    for (Point center : pores) {
        circle(img, center, radius, Scalar(0, 0, 255), 2);
    }

    // Display images
    namedWindow("Original Image", WINDOW_NORMAL);
    Imshow("binary", binary);
    Imshow("Original Image", img);
    namedWindow("Suppressed Image", WINDOW_NORMAL);
    Imshow("Suppressed Image", suppressed);
    cv::imwrite("pores.png", img);
    waitKey(0);

    return 0;
}
