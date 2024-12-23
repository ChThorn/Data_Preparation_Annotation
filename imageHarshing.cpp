#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <vector>

// Function to compute SSIM
double computeSSIM(const cv::Mat& img1, const cv::Mat& img2) {
    cv::Mat img1_gray, img2_gray;
    cv::cvtColor(img1, img1_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2, img2_gray, cv::COLOR_BGR2GRAY);

    cv::Mat img1_float, img2_float;
    img1_gray.convertTo(img1_float, CV_32F);
    img2_gray.convertTo(img2_float, CV_32F);

    cv::Mat mu1, mu2;
    cv::GaussianBlur(img1_float, mu1, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(img2_float, mu2, cv::Size(11, 11), 1.5);

    cv::Mat mu1_sq = mu1.mul(mu1);
    cv::Mat mu2_sq = mu2.mul(mu2);
    cv::Mat mu1_mu2 = mu1.mul(mu2);

    cv::Mat sigma1_sq, sigma2_sq, sigma12;
    cv::GaussianBlur(img1_float.mul(img1_float), sigma1_sq, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(img2_float.mul(img2_float), sigma2_sq, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(img1_float.mul(img2_float), sigma12, cv::Size(11, 11), 1.5);

    sigma1_sq -= mu1_sq;
    sigma2_sq -= mu2_sq;
    sigma12 -= mu1_mu2;

    double C1 = 6.5025, C2 = 58.5225;
    cv::Mat ssim_map = ((2 * mu1_mu2 + C1).mul(2 * sigma12 + C2)) /
                       ((mu1_sq + mu2_sq + C1).mul(sigma1_sq + sigma2_sq + C2));

    return cv::mean(ssim_map)[0];
}

int main() {
    std::string input_dir = "darknet_dataset_Capture/images/train"; // Set input directory path
    std::string output_dir = "darknet_dataset_Capture/images/harsh"; // Set output directory path

    std::filesystem::create_directory(output_dir); // Create output directory

    std::vector<cv::Mat> uniqueImages;
    std::vector<std::string> duplicates;

    double threshold = 0.95; // SSIM threshold for similarity

    for (const auto& entry : std::filesystem::directory_iterator(input_dir)) {
        cv::Mat img = cv::imread(entry.path().string());

        if (!img.empty()) {
            bool isDuplicate = false;

            for (const auto& storedImg : uniqueImages) {
                double ssim = computeSSIM(img, storedImg);
                if (ssim >= threshold) {
                    isDuplicate = true;
                    duplicates.push_back(entry.path().string());
                    break;
                }
            }

            if (!isDuplicate) {
                uniqueImages.push_back(img);
                std::string filename = entry.path().filename().string();
                cv::imwrite(output_dir + "/" + filename, img); // Save non-duplicate image
            }
        } else {
            std::cerr << "Could not read the image: " << entry.path() << std::endl;
        }
    }

    std::cout << "Duplicate images detected and removed: " << duplicates.size() << std::endl;
    for (const auto& dup : duplicates) {
        std::cout << dup << std::endl;
    }

    std::cout << "Duplicate removal completed." << std::endl;
    return 0;
}
