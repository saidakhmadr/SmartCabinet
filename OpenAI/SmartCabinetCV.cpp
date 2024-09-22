#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <map>
#include <ctime>

using namespace std;
using namespace cv;

map<string, bool> attendanceList; // To track which students are present

// Function to load student images
void loadStudentImages(const string& folderPath, vector<Mat>& images, vector<string>& names) {
    for (const auto& entry : filesystem::directory_iterator(folderPath)) {
        if (entry.is_regular_file()) {
            Mat img = imread(entry.path().string(), IMREAD_GRAYSCALE);
            if (!img.empty()) {
                images.push_back(img);
                names.push_back(entry.path().stem().string()); // Student ID as name
                attendanceList[entry.path().stem().string()] = false; // Mark initially as absent
            }
            else {
                cerr << "Could not load image: " << entry.path().string() << endl;
            }
        }
    }
}

// Function to save the attendance to an Excel (CSV) file
void saveAttendanceToCSV(const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file for writing: " << filename << endl;
        return;
    }

    file << "Student ID, Attendance\n"; // Correct newline
    for (const auto& entry : attendanceList) {
        file << entry.first << "," << (entry.second ? "Present" : "Absent") << "\n"; // Correct newline
    }
    file.close();
    cout << "Attendance saved to: " << filename << endl;
}

int main() {
    // Load Haar Cascade for face detection
    CascadeClassifier faceCascade;
    if (!faceCascade.load("haarcascade_frontalface_default.xml")) {
        cerr << "Error loading cascade!" << endl;
        return -1;
    }

    // Load student images
    vector<Mat> studentImages;
    vector<string> studentIDs;
    loadStudentImages("C:/Users/User/Pictures/Camera Roll", studentImages, studentIDs);

    if (studentImages.empty()) {
        cerr << "No student images found!" << endl;
        return -1;
    }

    // Initialize ORB detector
    Ptr<ORB> orb = ORB::create();
    vector<vector<KeyPoint>> studentKeypoints(studentImages.size());
    vector<Mat> studentDescriptors(studentImages.size());

    for (size_t i = 0; i < studentImages.size(); ++i) {
        orb->detectAndCompute(studentImages[i], noArray(), studentKeypoints[i], studentDescriptors[i]);
    }

    // Start capturing video from the webcam
    VideoCapture capture(1);
    if (!capture.isOpened()) {
        cerr << "Error opening camera!" << endl;
        return -1;
    }

    Mat frame;
    while (true) {
        capture >> frame;
        if (frame.empty()) {
            cerr << "Error getting frame!" << endl;
            break;
        }

        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // Detect faces
        vector<Rect> faces;
        faceCascade.detectMultiScale(gray, faces);

        for (const auto& face : faces) {
            Mat faceROI = gray(face);
            vector<KeyPoint> faceKeypoints;
            Mat faceDescriptors;

            // Compute ORB for the detected face
            orb->detectAndCompute(faceROI, noArray(), faceKeypoints, faceDescriptors);

            // Match the detected face with student images
            BFMatcher matcher(NORM_HAMMING, true);
            int bestMatchIndex = -1;
            double bestDistance = DBL_MAX;

            for (size_t i = 0; i < studentDescriptors.size(); ++i) {
                vector<DMatch> matches;
                matcher.match(faceDescriptors, studentDescriptors[i], matches);

                double totalDistance = 0;
                for (const auto& match : matches) {
                    totalDistance += match.distance;
                }

                if (totalDistance < bestDistance) {
                    bestDistance = totalDistance;
                    bestMatchIndex = i;
                }
            }

            // If a match is found, mark the student as present
            if (bestMatchIndex != -1 && bestDistance < 50) { // 50 is an arbitrary threshold
                string studentID = studentIDs[bestMatchIndex];
                attendanceList[studentID] = true; // Mark as present
                rectangle(frame, face, Scalar(0, 255, 0), 2);
                putText(frame, studentID, Point(face.x, face.y - 10), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
            }
            else {
                rectangle(frame, face, Scalar(0, 0, 255), 2);
                putText(frame, "Unknown", Point(face.x, face.y - 10), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
            }
        }

        // Display the video feed
        imshow("Attendance System", frame);

        // Capture keyboard input once
        char key = waitKey(30);
        if (key == 's') {
            saveAttendanceToCSV("attendance.csv");
        }
        if (key == 'q') break; // Break the loop if 'q' is pressed
    }

    capture.release();
    destroyAllWindows();
    return 0;
}
