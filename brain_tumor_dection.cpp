#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dlib/dnn.h>
#include <dlib/data_io.h>

using namespace std;
using namespace cv;

// Define the CNN model using Dlib's deep learning framework
using net_type = dlib::loss_multiclass_log<
    dlib::con<1, 5, 5, 1, 1, dlib::input<dlib::matrix<uint8_t>>>
>;

int main()
{
    // Load the pre-trained CNN model
    net_type net;
    dlib::deserialize("tumor_detection_model.dat") >> net;

    // Load and preprocess the input image
    Mat inputImage = imread("test_image.jpg", IMREAD_GRAYSCALE);
    if (inputImage.empty()) {
        cout << "Failed to load the input image!" << endl;
        return 1;
    }

    // Resize the input image to match the CNN model's input size
    cv::resize(inputImage, inputImage, cv::Size(64, 64));

    // Convert the image to Dlib's matrix format
    dlib::matrix<uint8_t> dlibImage;
    dlib::assign_image(dlibImage, dlib::cv_image<uint8_t>(inputImage));

    // Perform the tumor detection
    auto results = net(dlibImage);

    // Get the predicted label (tumor or non-tumor)
    int predictedLabel = dlib::index_of_max(results);

    // Display the prediction result
    if (predictedLabel == 0) {
        cout << "The image is classified as a brain tumor." << endl;
    }
    else {
        cout << "The image is classified as a non-tumor sample." << endl;
    }

    return 0;
}
