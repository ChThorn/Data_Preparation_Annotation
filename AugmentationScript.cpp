#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <thread> // For adding delay
#include <chrono> // For time-related functions

// Function to apply augmentations
std::vector<cv::Mat> augmentImage(const cv::Mat& img) {
    std::vector<cv::Mat> augmentedImages;

    // Original image
    augmentedImages.push_back(img);

    // Flip horizontally
    cv::Mat flipped;
    cv::flip(img, flipped, 1);
    augmentedImages.push_back(flipped);

    // Rotate by 30 degrees
    cv::Mat rotated;
    cv::Point2f center(img.cols / 2.0, img.rows / 2.0);
    cv::Mat rotMat = cv::getRotationMatrix2D(center, 30, 1.0);
    cv::warpAffine(img, rotated, rotMat, img.size());
    augmentedImages.push_back(rotated);

    // Adjust brightness
    cv::Mat bright;
    img.convertTo(bright, -1, 1, 50); // increase the brightness
    augmentedImages.push_back(bright);

    // Apply Gaussian blur
    cv::Mat blurred;
    cv::GaussianBlur(img, blurred, cv::Size(5, 5), 0);
    augmentedImages.push_back(blurred);

    // Scale image
    cv::Mat scaled;
    cv::resize(img, scaled, cv::Size(), 0.5, 0.5);
    augmentedImages.push_back(scaled);

    // Contrast adjustment
    cv::Mat contrast;
    img.convertTo(contrast, -1, 1.5, 0); // increase the contrast
    augmentedImages.push_back(contrast);

    // Add noise
    cv::Mat noise = cv::Mat(img.size(), img.type());
    cv::randn(noise, 0, 25); // Gaussian noise
    cv::Mat noisy = img + noise;
    augmentedImages.push_back(noisy);

    return augmentedImages;
}

int main() {
    std::string input_dir = "darknet_dataset_Capture/images/train"; // Set input directory path
    std::string output_dir = "darknet_dataset_Capture/images/trains"; // Set output directory path

    std::filesystem::create_directory(output_dir); // Create output directory

    for (const auto& entry : std::filesystem::directory_iterator(input_dir)) {
        cv::Mat img = cv::imread(entry.path().string());

        if (!img.empty()) {
            std::vector<cv::Mat> augmentedImages = augmentImage(img);

            for (size_t i = 0; i < augmentedImages.size(); ++i) {
                std::string filename = entry.path().stem().string() + "_aug_" + std::to_string(i) + ".jpg";
                cv::imwrite(output_dir + "/" + filename, augmentedImages[i]);
            }
        } else {
            std::cerr << "Could not read the image: " << entry.path() << std::endl;
        }
    }

    std::cout << "Data augmentation completed." << std::endl;
    return 0;
}
