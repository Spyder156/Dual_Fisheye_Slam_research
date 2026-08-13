/*
 * run_folder: ROS-free OpenVINS runner for an image-folder + imu.csv dataset
 * (project-local extension for the Insta360 dual-fisheye rig).
 *
 * Dataset layout (produced by scripts/extract_insv.py):
 *   <dir>/imu.csv      t,gx,gy,gz,ax,ay,az   (rad/s, m/s^2)
 *   <dir>/frames.csv   frame,t               (1-indexed, cam0 timeline)
 *   <dir>/cam0/%06d.jpg [cam1/%06d.jpg]
 *
 * Usage: run_folder <estimator_config.yaml> <dataset_dir> <out_traj.csv>
 */

#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "state/State.h"
#include "utils/opencv_yaml_parse.h"
#include "utils/sensor_data.h"

using namespace ov_msckf;

int main(int argc, char **argv) {

  if (argc != 4) {
    printf("usage: %s <estimator_config.yaml> <dataset_dir> <out_traj.csv>\n", argv[0]);
    return 1;
  }
  std::string config_path = argv[1];
  std::string dir = argv[2];
  std::string out_path = argv[3];

  auto parser = std::make_shared<ov_core::YamlParser>(config_path);
  std::string verbosity = "INFO";
  parser->parse_config("verbosity", verbosity, false);
  ov_core::Printer::setPrintLevel(verbosity);

  VioManagerOptions params;
  params.print_and_load(parser);
  auto app = std::make_shared<VioManager>(params);
  if (!parser->successful()) {
    printf(RED "unable to parse all parameters, please fix\n" RESET);
    return 1;
  }

  // ---- load imu.csv ----
  std::vector<ov_core::ImuData> imu;
  {
    std::ifstream f(dir + "/imu.csv");
    if (!f.is_open()) {
      printf("cannot open %s/imu.csv\n", dir.c_str());
      return 1;
    }
    std::string line;
    std::getline(f, line); // header
    while (std::getline(f, line)) {
      std::stringstream ss(line);
      std::string v;
      double d[7];
      for (int i = 0; i < 7; i++) {
        std::getline(ss, v, ',');
        d[i] = std::stod(v);
      }
      ov_core::ImuData m;
      m.timestamp = d[0];
      m.wm << d[1], d[2], d[3];
      m.am << d[4], d[5], d[6];
      imu.push_back(m);
    }
  }
  printf("loaded %zu imu samples [%.2f .. %.2f]\n", imu.size(), imu.front().timestamp, imu.back().timestamp);

  // ---- load frame timestamps ----
  std::vector<double> ftimes;
  {
    std::ifstream f(dir + "/frames.csv");
    if (!f.is_open()) {
      printf("cannot open %s/frames.csv\n", dir.c_str());
      return 1;
    }
    std::string line;
    std::getline(f, line); // header
    while (std::getline(f, line)) {
      std::stringstream ss(line);
      std::string a, b;
      std::getline(ss, a, ',');
      std::getline(ss, b, ',');
      ftimes.push_back(std::stod(b));
    }
  }
  int num_cams = params.state_options.num_cameras;
  printf("%zu frame timestamps, %d cameras\n", ftimes.size(), num_cams);

  std::ofstream out(out_path);
  out << "t,px,py,pz,qx,qy,qz,qw\n";

  // ---- main loop: feed imu, and pending camera frames once imu passes them ----
  size_t fi = 0;
  int fed = 0, logged = 0;
  for (const auto &m : imu) {
    app->feed_measurement_imu(m);
    while (fi < ftimes.size() && m.timestamp > ftimes[fi] + 0.05) {
      char name[32];
      snprintf(name, sizeof(name), "/%06zu.jpg", fi + 1);
      ov_core::CameraData cam;
      cam.timestamp = ftimes[fi];
      bool ok = true;
      for (int c = 0; c < num_cams; c++) {
        cv::Mat img = cv::imread(dir + "/cam" + std::to_string(c) + name, cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
          ok = false;
          break;
        }
        if (params.downsample_cameras)
          cv::resize(img, img, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
        cam.sensor_ids.push_back(c);
        cam.images.push_back(img);
        if (params.use_mask)
          cam.masks.push_back(params.masks.at(c));
        else
          cam.masks.push_back(cv::Mat::zeros(img.rows, img.cols, CV_8UC1));
      }
      if (ok) {
        app->feed_measurement_camera(cam);
        fed++;
        if (app->initialized()) {
          auto state = app->get_state();
          Eigen::Vector4d q = state->_imu->quat(); // JPL [x,y,z,w], R_GtoI
          Eigen::Vector3d p = state->_imu->pos();
          out << state->_timestamp << "," << p(0) << "," << p(1) << "," << p(2) << "," << q(0) << "," << q(1) << "," << q(2) << ","
              << q(3) << "\n";
          logged++;
        }
        if (fed % 100 == 0)
          printf("[%d/%zu] init=%d logged=%d\n", fed, ftimes.size(), (int)app->initialized(), logged);
      }
      fi++;
    }
  }
  out.close();
  printf("done: fed %d frames, logged %d poses -> %s\n", fed, logged, out_path.c_str());

  // final calibration states (useful with online calib enabled)
  if (app->initialized()) {
    auto state = app->get_state();
    for (int c = 0; c < num_cams; c++) {
      if (state->_cam_intrinsics.find(c) != state->_cam_intrinsics.end()) {
        std::cout << "cam" << c << " intrinsics: " << state->_cam_intrinsics.at(c)->value().transpose() << std::endl;
      }
      if (state->_calib_IMUtoCAM.find(c) != state->_calib_IMUtoCAM.end()) {
        std::cout << "cam" << c << " T_ItoC: " << state->_calib_IMUtoCAM.at(c)->value().transpose() << std::endl;
      }
    }
    if (state->_calib_dt_CAMtoIMU != nullptr) {
      std::cout << "t_off cam-imu: " << state->_calib_dt_CAMtoIMU->value()(0) << std::endl;
    }
  }
  return 0;
}
