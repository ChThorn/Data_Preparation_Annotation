#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <chrono>
#include <thread>

namespace fs = std::filesystem;

class DatasetCollector {
private:
    rs2::pipeline pipe;
    rs2::config cfg;
    std::string dataset_path;
    std::string images_path;
    std::string labels_path;
    int frame_count;
    int image_width;
    int image_height;
    std::vector<std::string> class_names;

public:
    DatasetCollector(const std::string& base_path, int width = 640, int height = 480) // width 640 height 480 
        : image_width(width), image_height(height) {
        // Create directory structure for darknet format
        dataset_path = base_path;
        images_path = dataset_path + "/images";
        labels_path = dataset_path + "/labels";

        createDirectories();
        setupRealSense();
        loadClassNames();

        // Find the highest frame number in existing files
        frame_count = findHighestFrameNumber();
        std::cout << "Starting from frame number: " << frame_count << std::endl; 
    }

    // New function to find the highest existing frame number
    int findHighestFrameNumber()
    {
        int highest = -1;

        // Check both train and valid directories
        std::vector<std::string> subsets = {"train", "valid"};
        for(const auto& subset : subsets)
        {
            std::string dir_path = images_path + "/" + subset;
            if(!fs::exists(dir_path)) continue;

            for(const auto& entry : fs::directory_iterator(dir_path))
            {
                if(entry.path().extension() == ".jpg")
                {
                    std::string filename = entry.path().stem().string();
                    try{
                        int num = std::stoi(filename);
                        highest = std::max(highest, num);
                    }
                    catch(...)
                    {
                        continue;
                    }
                }
            }
        }
        return highest +1;
    }

    void createDirectories() {
        // Create required directories
        fs::create_directories(images_path);
        fs::create_directories(labels_path);
        
        // Create train and valid directories
        fs::create_directories(images_path + "/train");
        fs::create_directories(images_path + "/valid");
        fs::create_directories(labels_path + "/train");
        fs::create_directories(labels_path + "/valid");
    }

    void setupRealSense() {
        cfg.enable_stream(RS2_STREAM_COLOR, image_width, image_height, RS2_FORMAT_BGR8, 30);
        pipe.start(cfg);
        
        // Warm up camera
        for(int i = 0; i < 30; i++) {
            pipe.wait_for_frames();
        }
    }

    void loadClassNames() {
        std::cout << "Enter class names (one per line, empty line to finish):\n";
        std::string class_name;
        while (std::getline(std::cin, class_name) && !class_name.empty()) {
            class_names.push_back(class_name);
        }

        // Save classes to obj.names
        std::ofstream names_file(dataset_path + "/obj.names");
        for (const auto& name : class_names) {
            names_file << name << "\n";
        }
        names_file.close();

        // Create obj.data file
        std::ofstream data_file(dataset_path + "/obj.data");
        data_file << "classes = " << class_names.size() << "\n";
        data_file << "train = " << dataset_path << "/train.txt\n";
        data_file << "valid = " << dataset_path << "/valid.txt\n";
        data_file << "names = " << dataset_path << "/obj.names\n";
        data_file << "backup = backup/\n";
        data_file.close();
    }

    cv::Mat captureFrame() {
        rs2::frameset frames = pipe.wait_for_frames();
        rs2::frame color_frame = frames.get_color_frame();
        return cv::Mat(cv::Size(image_width, image_height), CV_8UC3, 
                      (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
    }

    void collectDataset(int num_frames) {
        cv::namedWindow("Dataset Collection", cv::WINDOW_AUTOSIZE);
        std::cout << "Press 'SPACE' to capture, 'Q' to quit\n";

        while (frame_count < num_frames) {
            cv::Mat frame = captureFrame();
            
            // Show preview with overlay
            cv::Mat display = frame.clone();
            std::string info = "Captured: " + std::to_string(frame_count) + 
                             "/" + std::to_string(num_frames);
            cv::putText(display, info, cv::Point(10, 30), 
                       cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
            
            cv::imshow("Dataset Collection", display);
            char key = cv::waitKey(1);

            if (key == ' ') {  // Spacebar
                saveFrame(frame);
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
            else if (key == 'q') {
                break;
            }
        }
        
        cv::destroyAllWindows();
        createTrainValidLists();
    }

private:
    void saveFrame(const cv::Mat& frame) {
        // Determine if this frame goes to train or valid (80/20 split)
        std::string subset = (frame_count % 5 == 0) ? "valid" : "train";
        
        // Generate filename
        std::stringstream ss;
        ss << frame_count << ".jpg";
        std::string filename = ss.str();

        // Save image
        cv::imwrite(images_path + "/" + subset + "/" + filename, frame);

        // Create empty label file
        std::ofstream label_file(labels_path + "/" + subset + "/" + 
                               std::to_string(frame_count) + ".txt");
        label_file.close();

        frame_count++;
        std::cout << "Saved frame " << frame_count << " to " << subset << " set\n";
    }

    void createTrainValidLists() {
        // Create train.txt and valid.txt
        createImageList("train");
        createImageList("valid");
    }

    void createImageList(const std::string& subset) {
        std::string list_file = dataset_path + "/" + subset + ".txt";
        std::ofstream ofs(list_file);
        
        for (const auto& entry : fs::directory_iterator(images_path + "/" + subset)) {
            if (entry.path().extension() == ".jpg") {
                ofs << fs::absolute(entry.path()).string() << "\n";
            }
        }
        ofs.close();
    }

public:
    ~DatasetCollector() {
        pipe.stop();
    }
};

int main() {
    try {
        std::string dataset_path = "darknet_dataset_Capture";
        DatasetCollector collector(dataset_path);

        // Collect images
        int num_frames = 100;  // Change this to desired number of frames
        collector.collectDataset(num_frames);

        std::cout << "\nDataset collection complete. Next steps:\n";
        std::cout << "1. Use a labeling tool to annotate images\n";
        std::cout << "2. Verify train.txt and valid.txt files\n";
        std::cout << "3. Update obj.data if needed\n";
        std::cout << "4. Start training with darknet\n";

    } catch (const rs2::error& e) {
        std::cerr << "RealSense error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}