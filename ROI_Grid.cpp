#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>

cv::Rect roi;
bool drawing = false;
cv::Point start_point;

void mouseCallback(int event, int x, int y, int flags, void* userdata) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        drawing = true;
        start_point = cv::Point(x, y);
    } else if (event == cv::EVENT_MOUSEMOVE) {
        if (drawing) {
            roi = cv::Rect(start_point, cv::Point(x, y));
        }
    } else if (event == cv::EVENT_LBUTTONUP) {
        drawing = false;
    }
}

float get_depth_at_pixel(const rs2::depth_frame& frame, int x, int y) 
{
    // Get the depth value at the specified pixel
    uint16_t depth_value = frame.get_distance(x, y);
    return depth_value;
}


int main() {
    // Create a context and a pipeline
    rs2::context ctx;
    rs2::pipeline pipe(ctx);
    rs2::config cfg;

    // Configure the pipeline to enable the color and depth streams
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);

    // Start the pipeline
    rs2::pipeline_profile profile = pipe.start(cfg);

    // Create an OpenCV window to display the result
    const std::string window_name = "RealSense D456 ROI with Grid";
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback(window_name, mouseCallback);

    while (cv::waitKey(1) < 0) {
        // Wait for the next set of frames
        rs2::frameset frames = pipe.wait_for_frames();

        // Get color and depth frames
        rs2::video_frame color_frame = frames.get_color_frame();
        rs2::depth_frame depth_frame = frames.get_depth_frame();

        // Create OpenCV matrices from the color and depth frames
        cv::Mat color_image(cv::Size(640, 480), CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP);
        cv::Mat depth_image(cv::Size(640, 480), CV_16U, (void*)depth_frame.get_data(), cv::Mat::AUTO_STEP);

        // Draw the ROI if it is being defined
        if (!roi.empty()) {
            cv::rectangle(color_image, roi, cv::Scalar(0, 255, 0), 2);

            // Draw the grid within the ROI
            int rows = 2;
            int columns = 10;
            int cell_width = roi.width / columns;
            int cell_height = roi.height / rows;

            for (int i = 1; i < columns; ++i) {
                cv::line(color_image, cv::Point(roi.x + i * cell_width, roi.y), cv::Point(roi.x + i * cell_width, roi.y + roi.height), cv::Scalar(0, 255, 0), 1);
            }
            for (int j = 1; j < rows; ++j) {
                cv::line(color_image, cv::Point(roi.x, roi.y + j * cell_height), cv::Point(roi.x + roi.width, roi.y + j * cell_height), cv::Scalar(0, 255, 0), 1);
            }


            for (int i = 0; i < columns; ++i) 
            { 
                for (int j = 0; j < rows; ++j) 
                { 
                    int cell_x = roi.x + i * cell_width + cell_width / 2; 
                    int cell_y = roi.y + j * cell_height + cell_height / 2; 
                    float depth = get_depth_at_pixel(depth_frame, cell_x, cell_y); 
                    std::ostringstream depth_text; 
                    depth_text << std::fixed << std::setprecision(2) << depth << "m"; 
                    cv::putText(color_image, depth_text.str(), cv::Point(cell_x - 10, cell_y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1); 
                    // Optionally print depth values to the console 
                    std::cout << "Depth at (" << cell_x << ", " << cell_y << "): " << depth << "m" << std::endl; 
                    }
            }
        }

        // Display the result
        cv::imshow(window_name, color_image);
    }

    return 0;
}
