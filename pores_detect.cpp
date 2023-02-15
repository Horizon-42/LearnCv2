#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main() {
    // Load image and convert to grayscale
    Mat img = imread("image.jpg");
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

    // Threshold gradient magnitude to obtain binary image
    double threshold = 50.0;
    Mat binary;
    threshold(gradient, binary, threshold, 255, THRESH_BINARY);

    // Apply dilation and erosion to connect and shrink edges
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    Mat closed;
    morphologyEx(binary, closed, MORPH_CLOSE, kernel);

    // Apply non-maximum suppression to detect pores
    int radius = 15;
    Mat suppressed = Mat::zeros(closed.size(), closed.type());
    for (int y = radius; y < closed.rows - radius; y++) {
        for (int x = radius; x < closed.cols - radius; x++) {
            if (closed.at<uchar>(y, x) == 255) {
                Mat patch = closed(Rect(x - radius, y - radius, 2 * radius + 1, 2 * radius + 1));
                if (countNonZero(patch == 255) == 1) {
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
    imshow("Original Image", img);
    namedWindow("Suppressed Image", WINDOW_NORMAL);
    imshow("Suppressed Image", suppressed);
    waitKey(0);

    return 0;
}
