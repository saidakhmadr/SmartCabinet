#ifndef FACERECOGNITIONSYSTEM_H
#define FACERECOGNITIONSYSTEM_H

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <filesystem>
#include <map>
#include <string>

class FaceRecognitionSystem {
public:
    // Constructor
    FaceRecognitionSystem(const std::string& studentImagePath, const std::string& cascadePath);

    // Load student images
    void loadStudentImages();

    // Recognize faces from video feed
    void recognizeFromVideo();

    // Save attendance to CSV
    void saveAttendanceToCSV(const std::string& filename);

private:
    // Helper function to compute ORB descriptors
    void computeORBDescriptors(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);

    // Map of student names to their images
    std::map<std::string, std::vector<cv::Mat>> studentImages;
    std::map<std::string, bool> attendanceList;

    std::string studentImagePath; // Path to student images
    std::string cascadePath; // Path to face cascade file

    cv::Ptr<cv::ORB> orb; // ORB detector
    cv::CascadeClassifier faceCascade; // Haar Cascade for face detection
};

#endif
