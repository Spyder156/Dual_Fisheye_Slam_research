// Dump ELSED line segments for a folder of images.
//
// C++ rather than a python binding, because ELSED has to end up INSIDE
// ORB-SLAM3 in C++ anyway -- this tool exercises the exact library we will link
// there, and avoids a pybind11/python-dev dependency in the image.
//
// Output CSV: frame,x1,y1,x2,y2   (pixel endpoints; the spherical lifting and
// the angular-length filter happen downstream, where the camera model lives)
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "ELSED.h"

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "usage: line_extract <frames_dir> <list.txt> <out.csv> "
                 "[grad_thresh] [min_len_px]\n"
                 "  list.txt: one frame id per line (%06d.jpg)\n";
    return 1;
  }
  const std::string dir = argv[1], list = argv[2], out = argv[3];
  const int gradTh = (argc > 4) ? std::stoi(argv[4]) : 30;
  const int minLen = (argc > 5) ? std::stoi(argv[5]) : 15;

  upm::ELSEDParams p;
  p.gradientThreshold = gradTh;
  p.minLineLen = minLen;
  upm::ELSED elsed(p);

  std::ifstream f(list);
  std::ofstream o(out);
  o << "frame,x1,y1,x2,y2\n";
  std::string line;
  long nseg = 0, nimg = 0;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    const int id = std::stoi(line);
    char name[64];
    snprintf(name, sizeof(name), "/%06d.jpg", id);
    cv::Mat img = cv::imread(dir + name, cv::IMREAD_GRAYSCALE);
    if (img.empty()) continue;
    upm::Segments segs = elsed.detect(img);
    for (const auto& s : segs) {
      o << id << "," << s[0] << "," << s[1] << "," << s[2] << "," << s[3] << "\n";
      ++nseg;
    }
    ++nimg;
    if (nimg % 200 == 0) std::cerr << "  " << nimg << " images\r" << std::flush;
  }
  std::cerr << "\n" << nimg << " images, " << nseg << " segments -> " << out
            << "  (" << (nimg ? double(nseg) / nimg : 0) << " per image)\n";
  return 0;
}
