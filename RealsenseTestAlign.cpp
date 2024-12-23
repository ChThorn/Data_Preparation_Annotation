#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>

int main() {
    rs2::pipeline pipe;
    rs2::config cfg;
    
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    
    pipe.start(cfg);

    rs2::align align_to_color(RS2_STREAM_COLOR);

    while (true) {
        rs2::frameset frames = pipe.wait_for_frames();
        frames = align_to_color.process(frames);
        
        rs2::frame depth = frames.get_depth_frame();
        rs2::frame color = frames.get_color_frame();
        
        if (!depth || !color) continue;

        const int depth_width = depth.as<rs2::video_frame>().get_width();
        const int depth_height = depth.as<rs2::video_frame>().get_height();
        const int color_width = color.as<rs2::video_frame>().get_width();
        const int color_height = color.as<rs2::video_frame>().get_height();

        cv::Mat depth_image(cv::Size(depth_width, depth_height), CV_16UC1, (void*)depth.get_data(), cv::Mat::AUTO_STEP);
        cv::Mat color_image(cv::Size(color_width, color_height), CV_8UC3, (void*)color.get_data(), cv::Mat::AUTO_STEP);

        cv::Mat depth_colormap;
        cv::convertScaleAbs(depth_image, depth_colormap, 0.03);
        applyColorMap(depth_colormap, depth_colormap, cv::COLORMAP_JET);

        cv::Mat overlay;
        addWeighted(color_image, 0.7, depth_colormap, 0.3, 0, overlay);

        cv::imshow("RealSense Overlay", overlay);
        if (cv::waitKey(1) == 27) break;
    }

    return 0;
}
