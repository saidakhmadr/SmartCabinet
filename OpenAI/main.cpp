#include "SmartCabinetCV.h"

int main() {
    // Create an instance of FaceRecognitionSystem with the student images folder and cascade path
    FaceRecognitionSystem system("C:/Users/User/Pictures/Camera Roll", "haarcascade_frontalface_default.xml");

    // Load student images
    system.loadStudentImages();

    // Start recognizing faces from the video feed
    system.recognizeFromVideo();

    return 0;
}
