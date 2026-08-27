#include "Rig.h"

#include <iomanip>
#include <iostream>

namespace ORB_SLAM3 {

static const char* kTag = "[RIG]";

void Rig::Stage(const char* where, const char* what) {
    std::cout << kTag << " " << where << ": " << what << std::endl;
}

bool Rig::LoadFromSettings(const std::string& settingsPath) {
    cv::FileStorage fs(settingsPath, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        Stage("load", "settings file could not be opened -- rig DISABLED");
        return false;
    }
    if (fs["Rig.enabled"].empty() || (int)fs["Rig.enabled"] == 0) {
        Stage("load", "Rig.enabled absent or 0 -- running single-camera");
        mbEnabled = false;
        return false;
    }
    if (fs["Rig.T_c0_c1"].empty()) {
        // Fail loudly. A rig that silently falls back to identity would look
        // like a working rig and produce quietly wrong geometry.
        Stage("load", "FATAL: Rig.enabled=1 but Rig.T_c0_c1 is missing");
        exit(-1);
    }
    cv::Mat T;
    fs["Rig.T_c0_c1"] >> T;
    if (T.rows != 4 || T.cols != 4) {
        Stage("load", "FATAL: Rig.T_c0_c1 is not 4x4");
        exit(-1);
    }
    cv::Mat Tf;
    T.convertTo(Tf, CV_32F);
    Eigen::Matrix3f R;
    Eigen::Vector3f t;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) R(i, j) = Tf.at<float>(i, j);
        t(i) = Tf.at<float>(i, 3);
    }
    // re-orthonormalise: the yaml carries rounded values
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    R = svd.matrixU() * svd.matrixV().transpose();
    mT_c0_c1 = Sophus::SE3f(R, t);

    if (!fs["Rig.refine_rotation"].empty())
        mbRefineRotation = ((int)fs["Rig.refine_rotation"] != 0);
    if (!fs["Rig.refine_translation"].empty())
        mbRefineTranslation = ((int)fs["Rig.refine_translation"] != 0);

    mbEnabled = true;
    PrintSummary();
    return true;
}

float Rig::InterCameraAngleDeg() const {
    const Eigen::Vector3f f0(0.f, 0.f, 1.f);
    const Eigen::Vector3f f1 = mT_c0_c1.rotationMatrix() * f0;
    const float c = std::max(-1.f, std::min(1.f, f0.dot(f1)));
    return std::acos(c) * 180.f / static_cast<float>(M_PI);
}

float Rig::BaselineMetres() const { return mT_c0_c1.translation().norm(); }

void Rig::PrintSummary() const {
    std::cout << kTag << " ENABLED -- non-overlapping 2-camera rig, ref sensor cam0\n";
    std::cout << kTag << "   inter-camera angle : " << std::fixed << std::setprecision(3)
              << InterCameraAngleDeg() << " deg  (180.000 = ideal back-to-back)\n";
    std::cout << kTag << "   baseline           : " << std::setprecision(4)
              << BaselineMetres() * 100.f << " cm\n";
    std::cout << kTag << "   refine rotation    : " << (mbRefineRotation ? "yes" : "no") << "\n";
    std::cout << kTag << "   refine translation : " << (mbRefineTranslation ? "yes" : "no")
              << "   (default no: translation is weakly observable without overlap)\n";
    std::cout << kTag << " NOTE: rig is loaded and wired; residuals/tracking not yet"
              << " using it (staged rollout)." << std::endl;
}

}  // namespace ORB_SLAM3
