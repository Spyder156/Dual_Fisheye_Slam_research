// Convention probe for OKVIS2 + the Hilti dual-fisheye rig.
//
// Verifies, one assertion at a time, that what we *believe* about OKVIS2's
// camera model, extrinsics convention and FOV handling is what OKVIS2 actually
// does. Each check prints a value and an expectation so a wrong convention is
// visible rather than inferred.
//
// Reference values come from the official Hilti calibration and from our own
// independently validated Python implementation.

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <memory>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/imgcodecs.hpp>
#include <okvis/cameras/EquidistantDistortion.hpp>
#include <okvis/cameras/NCameraSystem.hpp>
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/kinematics/Transformation.hpp>

using Cam = okvis::cameras::PinholeCamera<okvis::cameras::EquidistantDistortion>;

static int g_pass = 0, g_fail = 0;

static void check(const char* name, double got, double want, double tol,
                  const char* unit = "") {
  const bool ok = std::abs(got - want) <= tol;
  std::printf("%-2d %-46s got %12.6f  want %12.6f %-5s  %s\n", g_pass + g_fail + 1,
              name, got, want, unit, ok ? "PASS" : "<<<< FAIL");
  ok ? ++g_pass : ++g_fail;
}

static void note(const char* name, const char* fmt, ...) {
  std::printf("   %-46s ", name);
  va_list a;
  va_start(a, fmt);
  std::vprintf(fmt, a);
  va_end(a);
  std::printf("\n");
}

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  // ---- official Hilti calibration (cam0) ----
  const double fx = 465.3015482593691, fy = 465.32303798346413;
  const double cx = 730.0455886686005, cy = 720.1427007671206;
  const double k1 = 0.025800718903376804, k2 = -0.010909240777406872;
  const double k3 = -0.0016899537986031076, k4 = 0.00014766801645260894;

  Cam cam0(1472, 1440, fx, fy, cx, cy,
           okvis::cameras::EquidistantDistortion(k1, k2, k3, k4));

  std::printf("=== A. CAMERA MODEL: does OKVIS2's equidistant match KB4 as we compute it?\n");

  // 1) optical axis must land on the principal point
  {
    Eigen::Vector2d px;
    cam0.project(Eigen::Vector3d(0, 0, 1), &px);
    check("axis ray -> cx", px[0], cx, 1e-6, "px");
    check("axis ray -> cy", px[1], cy, 1e-6, "px");
  }

  // 2) radius at known angles must equal f * theta_d (KB4 by hand)
  for (double deg : {10.0, 30.0, 60.0, 80.0}) {
    const double th = deg * M_PI / 180.0;
    const double thd =
        th * (1 + k1 * std::pow(th, 2) + k2 * std::pow(th, 4) +
              k3 * std::pow(th, 6) + k4 * std::pow(th, 8));
    Eigen::Vector2d px;
    cam0.project(Eigen::Vector3d(std::sin(th), 0, std::cos(th)), &px);
    char nm[64];
    std::snprintf(nm, sizeof(nm), "radius @ %.0f deg", deg);
    check(nm, px[0] - cx, fx * thd, 1e-3, "px");
  }

  // 3) round-trip: pixel -> bearing -> pixel
  for (double deg : {5.0, 45.0, 85.0}) {
    const double th = deg * M_PI / 180.0;
    const Eigen::Vector3d ray(std::sin(th), 0, std::cos(th));
    Eigen::Vector2d px;
    cam0.project(ray, &px);
    Eigen::Vector3d back;
    cam0.backProject(px, &back);
    back.normalize();
    const double ang = std::acos(std::min(1.0, back.dot(ray))) * 180.0 / M_PI;
    char nm[64];
    std::snprintf(nm, sizeof(nm), "round-trip error @ %.0f deg", deg);
    check(nm, ang, 0.0, 1e-6, "deg");
  }

  std::printf("\n=== B. FIELD OF VIEW: how much of the lens does OKVIS2 accept?\n");
  {
    double last_ok = 0;
    for (double deg = 1; deg <= 120; deg += 1.0) {
      const double th = deg * M_PI / 180.0;
      Eigen::Vector2d px;
      const auto st = cam0.project(
          Eigen::Vector3d(std::sin(th), 0, std::cos(th)), &px);
      if (st == okvis::cameras::ProjectionStatus::Successful) {
        last_ok = deg;
      }
    }
    note("max accepted incidence angle", "%.0f deg  (image circle reaches ~90 deg)", last_ok);
    check("FOV covers at least 80 deg", last_ok >= 80.0 ? 1.0 : 0.0, 1.0, 0.0);
  }

  std::printf("\n=== B2. CORNERS: how far off-axis does a CORNER pixel look?\n");
  {
    // The overlap test iterates over ALL pixels, including corners. For a
    // fisheye the corner is much further off-axis than the axis extremes.
    Eigen::Vector3d ray;
    const bool ok = cam0.backProject(Eigen::Vector2d(0, 0), &ray);
    ray.normalize();
    const double corner_deg = std::acos(std::min(1.0, ray[2])) * 180.0 / M_PI;
    note("corner pixel (0,0) incidence", "%.1f deg   %s", corner_deg,
         ok ? "(backProject OK)" : "(backProject FAILED)");
    note("radius to corner", "%.0f px  vs image-circle ~%.0f px",
         std::sqrt(cx * cx + cy * cy), cy);
    // If the corner exceeds half the inter-camera angle, the two frusta touch.
    check("corner stays inside own hemisphere", corner_deg < 90.0 ? 1.0 : 0.0, 1.0, 0.0);
  }

  std::printf("\n=== C. EXTRINSICS: is T_SC camera->IMU as we assumed?\n");
  // T_SC as written into our config = inverse(Hilti T_cam_imu)
  Eigen::Matrix4d T_SC0;
  T_SC0 << 0.0172144747722161, 0.9998263174555488, -0.0071412030913354, -0.0160046769358047,
      -0.0008034642120502, -0.0071284262145564, -0.9999742696614562, -0.0156241207009497,
      -0.9998514971252359, 0.0172197695390673, 0.0006806125511055, 0.0204108511207645,
      0, 0, 0, 1;
  Eigen::Matrix4d T_SC1;
  T_SC1 << 0.0011402115621430, 0.9999680618160693, 0.0079104529202893, -0.0146675657577340,
      0.0082939881547414, -0.0079196425876777, 0.9999342423488504, 0.0244024911836697,
      0.9999649542249365, -0.0010745273816893, -0.0083027533278688, 0.0195384403839513,
      0, 0, 0, 1;

  const Eigen::Vector3d C0 = T_SC0.block<3, 1>(0, 3);
  const Eigen::Vector3d C1 = T_SC1.block<3, 1>(0, 3);
  check("baseline |C0 - C1|", (C0 - C1).norm() * 100.0, 4.01, 0.05, "cm");

  const Eigen::Matrix3d R_rel =
      T_SC1.block<3, 3>(0, 0).transpose() * T_SC0.block<3, 3>(0, 0);
  const double ang = std::acos(std::min(1.0, (R_rel.trace() - 1) / 2)) * 180 / M_PI;
  check("cam0 -> cam1 rotation", ang, 179.56, 0.5, "deg");

  // Forward axes should be near-opposite for a back-to-back rig.
  const Eigen::Vector3d f0 = T_SC0.block<3, 3>(0, 0) * Eigen::Vector3d(0, 0, 1);
  const Eigen::Vector3d f1 = T_SC1.block<3, 3>(0, 0) * Eigen::Vector3d(0, 0, 1);
  check("angle between forward axes", std::acos(std::min(1.0, f0.dot(f1))) * 180 / M_PI,
        179.56, 0.5, "deg");

  std::printf("\n=== D. OVERLAP: does OKVIS2 detect that the cameras do not overlap?\n");
  {
    okvis::cameras::NCameraSystem nsys;
    auto g0 = std::make_shared<const Cam>(cam0);
    auto g1 = std::make_shared<const Cam>(cam0);
    nsys.addCamera(
        std::make_shared<const okvis::kinematics::Transformation>(okvis::kinematics::Transformation(T_SC0)),
        g0, okvis::cameras::NCameraSystem::DistortionType::Equidistant);
    nsys.addCamera(
        std::make_shared<const okvis::kinematics::Transformation>(okvis::kinematics::Transformation(T_SC1)),
        g1, okvis::cameras::NCameraSystem::DistortionType::Equidistant);
    nsys.computeOverlaps();
    const bool ov01 = nsys.hasOverlap(0, 1);
    const bool ov00 = nsys.hasOverlap(0, 0);
    note("hasOverlap(0,0) [self]", "%s  (expect true)", ov00 ? "true" : "false");
    note("hasOverlap(0,1) [cross]", "%s  (expect FALSE for back-to-back)",
         ov01 ? "true" : "false");
    check("cross-camera overlap correctly false", ov01 ? 1.0 : 0.0, 0.0, 0.0);

    // Which pixels does it think overlap? For a back-to-back rig the answer
    // should be none; anything here is an artefact.
    const cv::Mat& om = nsys.overlap(0, 1);
    const int n = cv::countNonZero(om);
    note("overlapping pixels cam1-seen-by-cam0", "%d of %d (%.3f%%)", n,
         om.rows * om.cols, 100.0 * n / (om.rows * om.cols));
    if (n > 0) {
      int minr = 1 << 30, maxr = 0;
      for (int v = 0; v < om.rows; ++v)
        for (int u = 0; u < om.cols; ++u)
          if (om.at<uchar>(v, u)) {
            const int dr = int(std::hypot(u - cx, v - cy));
            minr = std::min(minr, dr);
            maxr = std::max(maxr, dr);
          }
      note("their radius from principal point", "%d .. %d px (image circle ~%.0f px)",
           minr, maxr, cy);
      // Do those pixels backProject successfully at all?
      int bad = 0, tot = 0;
      for (int v = 0; v < om.rows; v += 7)
        for (int u = 0; u < om.cols; u += 7)
          if (om.at<uchar>(v, u)) {
            Eigen::Vector3d r;
            ++tot;
            if (!cam0.backProject(Eigen::Vector2d(u, v), &r)) ++bad;
          }
      note("of those, backProject FAILS for", "%d of %d sampled  <-- unchecked return value", bad, tot);
    }
  }

  std::printf("\n=== D2. OVERLAP WITH IMAGE-CIRCLE MASKS APPLIED\n");
  {
    // The rim pixels that survive the backProject fix still project into the
    // other camera because KB4 has no FOV limit and the image RECTANGLE is much
    // larger than the image CIRCLE. A validity mask is the real fix:
    // computeOverlaps only accepts ProjectionStatus::Successful, and a masked
    // pixel returns ProjectionStatus::Masked.
    auto m0 = cv::imread("/config/mask0.png", cv::IMREAD_GRAYSCALE);
    auto m1 = cv::imread("/config/mask1.png", cv::IMREAD_GRAYSCALE);
    if (m0.empty() || m1.empty()) {
      note("masks", "NOT FOUND at /config/mask{0,1}.png -- skipping");
    } else {
      auto c0 = std::make_shared<Cam>(cam0);
      auto c1 = std::make_shared<Cam>(cam0);
      c0->initialiseUndistortMaps(); c0->initialiseCameraAwarenessMaps();
      c1->initialiseUndistortMaps(); c1->initialiseCameraAwarenessMaps();
      check("mask0 applied", c0->setMask(m0) ? 1.0 : 0.0, 1.0, 0.0);
      check("mask1 applied", c1->setMask(m1) ? 1.0 : 0.0, 1.0, 0.0);
      note("valid pixels cam0", "%d of %d (%.1f%%)", cv::countNonZero(m0),
           m0.rows * m0.cols, 100.0 * cv::countNonZero(m0) / (m0.rows * m0.cols));
      note("marker","before NCameraSystem"); okvis::cameras::NCameraSystem ns;
      ns.addCamera(std::make_shared<const okvis::kinematics::Transformation>(
                       okvis::kinematics::Transformation(T_SC0)),
                   c0, okvis::cameras::NCameraSystem::DistortionType::Equidistant);
      ns.addCamera(std::make_shared<const okvis::kinematics::Transformation>(
                       okvis::kinematics::Transformation(T_SC1)),
                   c1, okvis::cameras::NCameraSystem::DistortionType::Equidistant);
      note("marker","cameras added, calling computeOverlaps"); ns.computeOverlaps(); note("marker","computeOverlaps returned");
      const int n = cv::countNonZero(ns.overlap(0, 1));
      note("overlapping pixels cam1-seen-by-cam0", "%d  (expect 0)", n);
      check("cross-camera overlap now FALSE", ns.hasOverlap(0, 1) ? 1.0 : 0.0, 0.0, 0.0);
    }
  }

  std::printf("\n=== E. GRAVITY / IMU FRAME sanity\n");
  {
    // A stationary IMU reads +g along its own up-axis. With T_SC0 the camera
    // "down" should map to roughly -up in the IMU frame; just report it so a
    // flipped convention is visible.
    const Eigen::Vector3d cam_down(0, 1, 0);  // +y is down in a camera frame
    const Eigen::Vector3d in_imu = T_SC0.block<3, 3>(0, 0) * cam_down;
    note("camera +y (down) in IMU frame", "[%+.3f %+.3f %+.3f]", in_imu[0], in_imu[1], in_imu[2]);
    note("=> IMU axis most aligned with DOWN", "%s",
         (std::abs(in_imu[0]) > std::abs(in_imu[1]) && std::abs(in_imu[0]) > std::abs(in_imu[2]))
             ? "x"
             : (std::abs(in_imu[1]) > std::abs(in_imu[2]) ? "y" : "z"));
  }

  std::printf("\n---- %d passed, %d FAILED ----\n", g_pass, g_fail);
  return g_fail == 0 ? 0 : 1;
}
