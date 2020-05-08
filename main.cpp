#include <iostream>
#include <fstream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/calib3d/calib3d.hpp"

#include "opencv2/calib3d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

using namespace cv;
using namespace std;

static void saveXYZ(const char* filename, const Mat& mat)
{
    const double max_z = 1.0e4;
    FILE* fp = fopen(filename, "wt");
    for(int y = 0; y < mat.rows; y++)
    {
        for(int x = 0; x < mat.cols; x++)
        {
            Vec3f point = mat.at<Vec3f>(y, x);
            if(fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
            fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);
        }
    }
    fclose(fp);
}

string type2str(int type) {
    string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch ( depth ) {
        case CV_8U:  r = "8U"; break;
        case CV_8S:  r = "8S"; break;
        case CV_16U: r = "16U"; break;
        case CV_16S: r = "16S"; break;
        case CV_32S: r = "32S"; break;
        case CV_32F: r = "32F"; break;
        case CV_64F: r = "64F"; break;
        default:     r = "User"; break;
    }

    r += "C";
    r += (chans+'0');

    return r;
}

void printRow(const string name, const cv::Mat m) {
    std::cout << name << ": (" << m.size()  <<  "), type:" <<   type2str(m.type()) << " = " << std::endl << m.row(0) << std::endl << std::endl;
}

void print(const string name, const cv::Mat m) {
    std::cout << name << ": ( " << m.size()  <<  " ), type: " <<   m.type() << " = " << std::endl << m << std::endl << std::endl;
//    std::cout << name << ": ( " << m.size()  <<  " ), type: " <<   m.type() << " = " << std::endl << FormattedMat(m) << std::endl << std::endl;
}

int main() {
    std::string img1_filename = "coneLeft.png";
    std::string img2_filename = "coneRight.png";

    int color_mode = 0; //CV_LOAD_IMAGE_GRAYSCALE
    cv::Mat img1 = cv::imread(img1_filename, color_mode);
    cv::Mat img2 = cv::imread(img2_filename, color_mode);
    printRow("img1", img1);
    printRow("img2", img2);

    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;


    auto featuresSIFT = cv::xfeatures2d::SIFT::create();
    featuresSIFT->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
    featuresSIFT->detectAndCompute(img2, noArray(), keypoints2, descriptors2);
    printRow("descriptors1", descriptors1);
    printRow("descriptors2", descriptors2);

    // Use KNN to find 2 matches for each point so we can apply the ratio test from the original
    // SIFT paper (https://people.eecs.berkeley.edu/~malik/cs294/lowe-ijcv04.pdf)
    std::vector<std::vector<cv::DMatch>> rawMatches;
    std::vector<cv::DMatch> goodMatches;

    auto matcher = cv::BFMatcher::create();
    matcher->knnMatch(descriptors1, descriptors2, rawMatches, 2);
    for (const auto& matchPair : rawMatches) {
        if (matchPair[0].distance < 0.75 * matchPair[1].distance) {
            goodMatches.push_back(matchPair[0]);
        }
    }

    // Create image with lines drawn between matched points. As we iterate through each point, log
    // its info
    cv::Mat combinedSrc;
    cv::hconcat(img1, img2, combinedSrc);
    cv::cvtColor(combinedSrc, combinedSrc, cv::COLOR_GRAY2RGB);
    printRow("combinedSrc", combinedSrc);

    std::vector<cv::Point2f> sourceKeypoints, destinationKeypoints;

    std::stringstream ss;
    ss << "\nMatches:";
    cv::RNG rng(12345);
    for (const auto& goodMatch : goodMatches) {
        cv::KeyPoint k1 = keypoints1[goodMatch.queryIdx];
        cv::KeyPoint k2 = keypoints2[goodMatch.trainIdx];

        sourceKeypoints.push_back(k1.pt);
        destinationKeypoints.push_back(k2.pt);

        int xOffset = img1.cols;
        cv::line(combinedSrc, k1.pt,
                 cv::Point2f(k2.pt.x + xOffset, k2.pt.y),
                 cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)));
        ss << "\n queryIdx:" << goodMatch.queryIdx << " trainIdx:" << goodMatch.trainIdx
           << " distance:" << goodMatch.distance;
    }

//    std::cout << ss.str() << std::endl;
    std::cout << "goodMatches.size():" << goodMatches.size() << std::endl;
    string combinedPath = "matched.png";
    cv::imwrite(combinedPath, combinedSrc);
    printRow("combinedSrc", combinedSrc);

    // collect samples with (1 - p)^e
    std::vector<cv::Point2f> pointsTranslateA, pointsTranslateB;
    for (const auto& match: goodMatches) {
        pointsTranslateA.emplace_back(keypoints1[match.queryIdx].pt);
        pointsTranslateB.emplace_back(keypoints2[match.trainIdx].pt);
    }

    cv::Mat matrixF = cv::findFundamentalMat(pointsTranslateA, pointsTranslateB,
            FM_RANSAC, 3, 0.99);
//    std::vector<uint8_t> inliersMask(goodMatches.size());
//    cv::Mat matrixF = cv::findFundamentalMat(sourceKeypoints, destinationKeypoints, inliersMask);
//    cv::Mat matrixF = cv::findFundamentalMat(sourceKeypoints, destinationKeypoints, FM_8POINT);

    cv::Mat H1(4,4, img1.type());
    cv::Mat H2(4,4, img2.type());
    cv::stereoRectifyUncalibrated(sourceKeypoints, destinationKeypoints, matrixF, img1.size(), H1, H2);
    print("H1", H1);
    print("H2", H2);

    cv::Mat rectified1(img1.size(), img1.type());
    cv::warpPerspective(img1, rectified1, H1, img1.size());
    cv::imwrite("rectified1.png", rectified1);

    cv::Mat rectified2(img2.size(), img2.type());
    cv::warpPerspective(img2, rectified2, H2, img2.size());
    cv::imwrite("rectified2.png", rectified2);

    printRow("rectified1", rectified1);
    printRow("rectified2", rectified2);

    Ptr<StereoSGBM> sgbm = StereoSGBM::create(0,16,3);
    int sgbmWinSize = 3;

    int cn = img1.channels();
    int minDisparity = 16;
    int numberOfDisparities = 112 - minDisparity;

    sgbm->setMinDisparity(minDisparity);
    sgbm->setNumDisparities(numberOfDisparities);
    sgbm->setBlockSize(sgbmWinSize);
    sgbm->setUniquenessRatio(10);
    sgbm->setSpeckleWindowSize(100);
    sgbm->setSpeckleRange(32);
    sgbm->setDisp12MaxDiff(1);

    sgbm->setP1(8*cn*sgbmWinSize*sgbmWinSize);
    sgbm->setP2(32*cn*sgbmWinSize*sgbmWinSize);
    sgbm->setPreFilterCap(20);
//    sgbm->setMode(false);

    cv::Mat disp, disp8;
    float disparity_multiplier = 1.0f;
    sgbm->compute(img1, img2, disp);
    if (disp.type() == CV_16S)
        disparity_multiplier = 16.0f;
    disp.convertTo(disp8, CV_8U, 255/(numberOfDisparities*16.));
    namedWindow("disparity", 0);
    imshow("disparity", disp8);

    Size img_size = img1.size();
    float w = img_size.width;
    float h = img_size.height;
    float focal_length = 10;

    cv::Mat Q = (Mat_<float>(4,4) <<
            1, 0, 0, w / 2.0,
            0, -1, 0, h / 2.0,
            0, 0, focal_length, 0,
            0, 0, 0, 1);
    print("Q", Q);

    Mat xyz;
    Mat floatDisp;
    disp.convertTo(floatDisp, CV_32F, 1.0f / disparity_multiplier);
    reprojectImageTo3D(, xyz, Q, true);
//    cv::cvtColor(img1, img1, cv::COLOR_GRAY2RGB);
    saveXYZ("pointcloud.xyz", xyz);
    print("floatDisp", floatDisp);

}
