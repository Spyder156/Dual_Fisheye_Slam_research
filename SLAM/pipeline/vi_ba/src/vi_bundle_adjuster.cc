// Visual-inertial bundle adjustment for COLMAP using Basalt's recovered
// non-linear factors (Usenko et al., RA-L 2019).
//
// Takes a triangulated reconstruction (poses initialized from Basalt VIO) and
// jointly minimizes:
//   - reprojection errors (COLMAP's default bundle adjuster), plus
//   - relative-pose factors between keyframe IMU frames, and
//   - roll-pitch (gravity) factors per keyframe,
// both recovered from Basalt's marginalization priors and whitened with their
// information matrices.
//
// The inertial residuals replicate Basalt's definitions exactly (decoupled
// SE(3) log, IMU frame, [translation; rotation] ordering), evaluated through
// the camera poses via the fixed T_imu_cam extrinsic, so the exported
// information matrices apply without any convention conversion.
//
// Factors come as a plain-text file produced by tools/factors_json_to_txt.py.

#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/ceres.h>

#include "colmap/estimators/bundle_adjustment.h"
#include "colmap/estimators/bundle_adjustment_ceres.h"
#include "colmap/estimators/cost_functions/quaternion_utils.h"
#include "colmap/estimators/cost_functions/utils.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/logging.h"

namespace {

using Mat6 = Eigen::Matrix<double, 6, 6>;
using Mat2 = Eigen::Matrix2d;

struct RelPoseFactor {
  int64_t t_i_ns, t_j_ns;
  Eigen::Quaterniond q_ij;  // T_i_j: imu_j -> imu_i
  Eigen::Vector3d t_ij;
  Mat6 sqrt_info;  // upper-triangular L^T with cov_inv = L L^T
};

struct RollPitchFactor {
  int64_t t_ns;
  Eigen::Quaterniond q_w_i_meas;
  Mat2 sqrt_info;
};

// Residual: sqrt_info * se3_logd(T_i_j_prior * imu_j_from_w * w_from_imu_i),
// with se3_logd = [translation; so3_log], imu_from_world = C * cam_from_world.
// Parameter blocks: cam_i_from_world, cam_j_from_world (colmap 7-param
// [qx qy qz qw tx ty tz]).
struct BasaltRelPoseCostFunctor {
  BasaltRelPoseCostFunctor(const RelPoseFactor& f, const Eigen::Quaterniond& q_C,
                           const Eigen::Vector3d& t_C)
      : f_(f), q_C_(q_C), t_C_(t_C) {}

  template <typename T>
  bool operator()(const T* const i_fw, const T* const j_fw, T* res) const {
    using Quat = Eigen::Quaternion<T>;
    using Vec3 = Eigen::Matrix<T, 3, 1>;

    const Eigen::Map<const Quat> q_cw_i(i_fw);
    const Eigen::Map<const Vec3> t_cw_i(i_fw + 4);
    const Eigen::Map<const Quat> q_cw_j(j_fw);
    const Eigen::Map<const Vec3> t_cw_j(j_fw + 4);

    // imu_from_world = C o cam_from_world
    const Quat q_C = q_C_.cast<T>();
    const Vec3 t_C = t_C_.cast<T>();
    const Quat q_iw_i = q_C * q_cw_i;
    const Vec3 t_iw_i = q_C * t_cw_i + t_C;
    const Quat q_iw_j = q_C * q_cw_j;
    const Vec3 t_iw_j = q_C * t_cw_j + t_C;

    // T_j_i_est = imu_j_from_world o (imu_i_from_world)^-1
    const Quat q_ji = q_iw_j * q_iw_i.conjugate();
    const Vec3 t_ji = t_iw_j - q_ji * t_iw_i;

    // M = T_i_j_prior o T_j_i_est  (identity if estimate matches prior)
    const Quat q_m = f_.q_ij.cast<T>() * q_ji;
    const Vec3 t_m = f_.q_ij.cast<T>() * t_ji + f_.t_ij.cast<T>();

    Eigen::Matrix<T, 6, 1> r;
    r.template head<3>() = t_m;
    T aa[3];
    // ceres-style: angle-axis from quaternion coeffs (x, y, z, w)
    colmap::AngleAxisFromEigenQuaternion(q_m.coeffs().data(), aa);
    r[3] = aa[0];
    r[4] = aa[1];
    r[5] = aa[2];

    Eigen::Map<Eigen::Matrix<T, 6, 1>> res_map(res);
    res_map = f_.sqrt_info.cast<T>() * r;
    return true;
  }

  static ceres::CostFunction* Create(const RelPoseFactor& f,
                                     const Eigen::Quaterniond& q_C,
                                     const Eigen::Vector3d& t_C) {
    return new ceres::AutoDiffCostFunction<BasaltRelPoseCostFunctor, 6, 7, 7>(
        new BasaltRelPoseCostFunctor(f, q_C, t_C));
  }

  RelPoseFactor f_;
  Eigen::Quaterniond q_C_;
  Eigen::Vector3d t_C_;
};

using Mat9 = Eigen::Matrix<double, 9, 9>;

// One preintegrated IMU interval between two keyframes (body/IMU frame).
struct ImuPreintFactor {
  int64_t t_i_ns, t_j_ns;
  double dt;
  Eigen::Quaterniond dq;      // preintegrated rotation  i -> j
  Eigen::Vector3d dv, dp;     // preintegrated velocity / position deltas
  Eigen::Matrix3d J_q_bg, J_v_bg, J_v_ba, J_p_bg, J_p_ba;  // bias Jacobians
  Mat9 sqrt_info;             // whitening for [dp; dv; dq]
};

// Residual (9): standard visual-inertial preintegration error, expressed in
// body frame i, with gravity FIXED along -Z in the (gravity-aligned) world:
//   r_p = R_i^T (p_j - p_i - v_i dt - 1/2 g dt^2) - (dp + J_p_ba ba + J_p_bg bg)
//   r_v = R_i^T (v_j - v_i - g dt)                - (dv + J_v_ba ba + J_v_bg bg)
//   r_q = Log( (dq * Exp(J_q_bg bg))^-1 * R_i^T R_j )
// Parameter blocks: cam_i_from_world(7), cam_j_from_world(7),
//                   v_i(3), v_j(3), bias(6) = [ba(3); bg(3)]
struct ImuPreintCostFunctor {
  ImuPreintCostFunctor(const ImuPreintFactor& f, const Eigen::Quaterniond& q_C,
                       const Eigen::Vector3d& t_C, double gravity)
      : f_(f), q_C_(q_C), t_C_(t_C), g_(gravity) {}

  template <typename T>
  bool operator()(const T* const i_fw, const T* const j_fw, const T* const v_i,
                  const T* const v_j, const T* const bias, T* res) const {
    using Quat = Eigen::Quaternion<T>;
    using Vec3 = Eigen::Matrix<T, 3, 1>;

    const Eigen::Map<const Quat> q_cw_i(i_fw);
    const Eigen::Map<const Vec3> t_cw_i(i_fw + 4);
    const Eigen::Map<const Quat> q_cw_j(j_fw);
    const Eigen::Map<const Vec3> t_cw_j(j_fw + 4);
    const Quat q_C = q_C_.cast<T>();
    const Vec3 t_C = t_C_.cast<T>();

    // imu_from_world, then invert -> world_from_imu (R_i, p_i)
    const Quat q_iw_i = q_C * q_cw_i;
    const Vec3 t_iw_i = q_C * t_cw_i + t_C;
    const Quat q_iw_j = q_C * q_cw_j;
    const Vec3 t_iw_j = q_C * t_cw_j + t_C;
    const Quat R_i = q_iw_i.conjugate();
    const Vec3 p_i = -(R_i * t_iw_i);
    const Quat R_j = q_iw_j.conjugate();
    const Vec3 p_j = -(R_j * t_iw_j);

    const Eigen::Map<const Vec3> ba(bias);
    const Eigen::Map<const Vec3> bg(bias + 3);
    const Eigen::Map<const Vec3> vi(v_i);
    const Eigen::Map<const Vec3> vj(v_j);

    const T dt = T(f_.dt);
    Vec3 g_w(T(0), T(0), T(-g_));

    const Vec3 dp_c = f_.dp.cast<T>() + f_.J_p_ba.cast<T>() * ba +
                      f_.J_p_bg.cast<T>() * bg;
    const Vec3 dv_c = f_.dv.cast<T>() + f_.J_v_ba.cast<T>() * ba +
                      f_.J_v_bg.cast<T>() * bg;

    Eigen::Matrix<T, 9, 1> r;
    r.template segment<3>(0) =
        R_i.conjugate() * (p_j - p_i - vi * dt - T(0.5) * g_w * dt * dt) - dp_c;
    r.template segment<3>(3) =
        R_i.conjugate() * (vj - vi - g_w * dt) - dv_c;

    // rotation: dq corrected by gyro bias, compared to R_i^T R_j
    const Vec3 dtheta = f_.J_q_bg.cast<T>() * bg;
    const T ang = dtheta.norm();
    Quat dq_bias;
    if (ang > T(1e-9)) {
      const Vec3 ax = dtheta / ang;
      dq_bias = Quat(Eigen::AngleAxis<T>(ang, ax));
    } else {
      dq_bias = Quat(T(1), T(0), T(0), T(0));
    }
    const Quat dq_corr = f_.dq.cast<T>() * dq_bias;
    const Quat q_err = dq_corr.conjugate() * (R_i.conjugate() * R_j);
    T aa[3];
    colmap::AngleAxisFromEigenQuaternion(q_err.coeffs().data(), aa);
    r[6] = aa[0];
    r[7] = aa[1];
    r[8] = aa[2];

    Eigen::Map<Eigen::Matrix<T, 9, 1>> res_map(res);
    res_map = f_.sqrt_info.cast<T>() * r;
    return true;
  }

  static ceres::CostFunction* Create(const ImuPreintFactor& f,
                                     const Eigen::Quaterniond& q_C,
                                     const Eigen::Vector3d& t_C, double g) {
    return new ceres::AutoDiffCostFunction<ImuPreintCostFunctor, 9, 7, 7, 3, 3,
                                           6>(
        new ImuPreintCostFunctor(f, q_C, t_C, g));
  }

  ImuPreintFactor f_;
  Eigen::Quaterniond q_C_;
  Eigen::Vector3d t_C_;
  double g_;
};

// Residual: sqrt_info * ((R_w_i_meas * R_imu_from_world_est) * (-e_z)).head<2>
// Constrains roll/pitch (gravity direction) of the IMU frame; exact replica of
// basalt::rollPitchError.
struct BasaltRollPitchCostFunctor {
  BasaltRollPitchCostFunctor(const RollPitchFactor& f,
                             const Eigen::Quaterniond& q_C)
      : q_meas_C_(f.q_w_i_meas * q_C), sqrt_info_(f.sqrt_info) {}

  template <typename T>
  bool operator()(const T* const cam_fw, T* res) const {
    using Quat = Eigen::Quaternion<T>;
    using Vec3 = Eigen::Matrix<T, 3, 1>;
    const Eigen::Map<const Quat> q_cw(cam_fw);
    // R_meas * R_imu_from_world = (q_meas * q_C) * q_cw
    const Vec3 v = q_meas_C_.cast<T>() * (q_cw * Vec3(T(0), T(0), T(-1)));
    Eigen::Map<Eigen::Matrix<T, 2, 1>> r(res);
    r = sqrt_info_.cast<T>() * v.template head<2>();
    return true;
  }

  static ceres::CostFunction* Create(const RollPitchFactor& f,
                                     const Eigen::Quaterniond& q_C) {
    return new ceres::AutoDiffCostFunction<BasaltRollPitchCostFunctor, 2, 7>(
        new BasaltRollPitchCostFunctor(f, q_C));
  }

  Eigen::Quaterniond q_meas_C_;
  Mat2 sqrt_info_;
};

template <int N>
Eigen::Matrix<double, N, N> SqrtInfo(const Eigen::Matrix<double, N, N>& info) {
  Eigen::Matrix<double, N, N> sym = 0.5 * (info + info.transpose());
  Eigen::LLT<Eigen::Matrix<double, N, N>> llt(sym);
  if (llt.info() != Eigen::Success) {
    sym += 1e-9 * Eigen::Matrix<double, N, N>::Identity();
    llt.compute(sym);
  }
  return llt.matrixU();  // U with U^T U = info; residual whitened by U
}

}  // namespace

int main(int argc, char** argv) {
  std::string input_path, output_path, factors_path;
  bool no_inertial = false;
  int max_iterations = 100;
  double inertial_weight = 1.0;
  double factor_huber = 0.0;  // Huber threshold on whitened factor residuals
                              // (in sigmas); 0 = disabled
  std::string imu_preint_path;  // optional preintegrated-IMU factor file
  double gravity = 9.805;       // |g|, world assumed gravity-aligned (-Z)

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--input_path") input_path = argv[++i];
    else if (arg == "--output_path") output_path = argv[++i];
    else if (arg == "--factors") factors_path = argv[++i];
    else if (arg == "--no_inertial") no_inertial = true;
    else if (arg == "--max_iterations") max_iterations = std::stoi(argv[++i]);
    else if (arg == "--inertial_weight") inertial_weight = std::stod(argv[++i]);
    else if (arg == "--factor_huber") factor_huber = std::stod(argv[++i]);
    else if (arg == "--imu_preint") imu_preint_path = argv[++i];
    else if (arg == "--gravity") gravity = std::stod(argv[++i]);
    else {
      std::cerr << "unknown arg: " << arg << std::endl;
      return 1;
    }
  }
  THROW_CHECK(!input_path.empty() && !output_path.empty() &&
              !factors_path.empty());

  colmap::Reconstruction recon;
  recon.Read(input_path);
  LOG(INFO) << "Loaded reconstruction: " << recon.NumRegImages() << " images, "
            << recon.NumPoints3D() << " points";

  // ---- load factors ----
  std::ifstream fs(factors_path);
  THROW_CHECK(fs.is_open()) << factors_path;
  std::string tag;
  Eigen::Quaterniond q_C;
  Eigen::Vector3d t_C;
  fs >> tag >> t_C.x() >> t_C.y() >> t_C.z() >> q_C.x() >> q_C.y() >> q_C.z() >>
      q_C.w();
  THROW_CHECK_EQ(tag, "T_IMU_CAM");

  size_t n_rel, n_rp;
  std::vector<RelPoseFactor> rel_factors;
  std::vector<RollPitchFactor> rp_factors;
  fs >> tag >> n_rel;
  THROW_CHECK_EQ(tag, "RELPOSE");
  for (size_t k = 0; k < n_rel; ++k) {
    RelPoseFactor f;
    Mat6 info;
    fs >> f.t_i_ns >> f.t_j_ns >> f.t_ij.x() >> f.t_ij.y() >> f.t_ij.z() >>
        f.q_ij.x() >> f.q_ij.y() >> f.q_ij.z() >> f.q_ij.w();
    for (int r = 0; r < 6; ++r)
      for (int c = 0; c < 6; ++c) fs >> info(r, c);
    f.sqrt_info = SqrtInfo<6>(inertial_weight * info);
    rel_factors.push_back(f);
  }
  fs >> tag >> n_rp;
  THROW_CHECK_EQ(tag, "ROLLPITCH");
  for (size_t k = 0; k < n_rp; ++k) {
    RollPitchFactor f;
    Mat2 info;
    fs >> f.t_ns >> f.q_w_i_meas.x() >> f.q_w_i_meas.y() >> f.q_w_i_meas.z() >>
        f.q_w_i_meas.w();
    for (int r = 0; r < 2; ++r)
      for (int c = 0; c < 2; ++c) fs >> info(r, c);
    f.sqrt_info = SqrtInfo<2>(inertial_weight * info);
    rp_factors.push_back(f);
  }
  LOG(INFO) << "Loaded " << rel_factors.size() << " rel-pose + "
            << rp_factors.size() << " roll-pitch factors";

  std::vector<ImuPreintFactor> imu_factors;
  std::map<int64_t, std::array<double, 3>> imu_init_vel;
  std::array<double, 6> imu_init_bias = {0, 0, 0, 0, 0, 0};
  if (!imu_preint_path.empty()) {
    std::ifstream ims(imu_preint_path);
    THROW_CHECK(ims.is_open()) << imu_preint_path;
    std::string itag;
    size_t n_imu = 0;
    ims >> itag >> n_imu;
    THROW_CHECK_EQ(itag, "IMUPREINT");
    for (size_t k = 0; k < n_imu; ++k) {
      ImuPreintFactor f;
      ims >> f.t_i_ns >> f.t_j_ns >> f.dt;
      ims >> f.dq.x() >> f.dq.y() >> f.dq.z() >> f.dq.w();
      ims >> f.dv.x() >> f.dv.y() >> f.dv.z();
      ims >> f.dp.x() >> f.dp.y() >> f.dp.z();
      auto read33 = [&ims](Eigen::Matrix3d& M) {
        for (int r = 0; r < 3; ++r)
          for (int c = 0; c < 3; ++c) ims >> M(r, c);
      };
      read33(f.J_q_bg);
      read33(f.J_v_bg);
      read33(f.J_v_ba);
      read33(f.J_p_bg);
      read33(f.J_p_ba);
      Mat9 info;
      for (int r = 0; r < 9; ++r)
        for (int c = 0; c < 9; ++c) ims >> info(r, c);
      f.sqrt_info = SqrtInfo<9>(inertial_weight * info);
      imu_factors.push_back(f);
    }
    LOG(INFO) << "Loaded " << imu_factors.size() << " IMU preintegration factors";
    // optional init block: velocities per keyframe + initial bias
    std::string btag;
    if (ims >> btag && btag == "IMUINIT") {
      size_t n_init = 0;
      ims >> n_init >> imu_init_bias[0] >> imu_init_bias[1] >>
          imu_init_bias[2] >> imu_init_bias[3] >> imu_init_bias[4] >>
          imu_init_bias[5];
      for (size_t k = 0; k < n_init; ++k) {
        int64_t t_ns;
        std::array<double, 3> v;
        ims >> t_ns >> v[0] >> v[1] >> v[2];
        imu_init_vel[t_ns] = v;
      }
      LOG(INFO) << "Loaded init: " << imu_init_vel.size() << " velocities, bias ["
                << imu_init_bias[0] << ", " << imu_init_bias[1] << ", "
                << imu_init_bias[2] << "]";
    }
  }

  // ---- map timestamps to frames ----
  std::map<int64_t, colmap::Frame*> t_to_frame;
  std::map<int64_t, colmap::frame_t> t_to_frame_id;
  for (const auto& [image_id, image] : recon.Images()) {
    std::string stem =
        image.Name().substr(0, image.Name().find_last_of('.'));
    const size_t slash = stem.find_last_of('/');
    if (slash != std::string::npos) stem = stem.substr(slash + 1);
    const int64_t t_ns = std::stoll(stem);
    t_to_frame[t_ns] = image.FramePtr();
    t_to_frame_id[t_ns] = image.FrameId();
  }

  // ---- configure BA ----
  colmap::BundleAdjustmentOptions options;
  options.refine_focal_length = false;
  options.refine_principal_point = false;
  options.refine_extra_params = false;
  options.ceres->loss_function_type =
      colmap::CeresBundleAdjustmentOptions::LossFunctionType::SOFT_L1;
  options.ceres->loss_function_scale = 1.0;
  options.ceres->solver_options.max_num_iterations = max_iterations;
  options.ceres->solver_options.minimizer_progress_to_stdout = false;

  colmap::BundleAdjustmentConfig config;
  for (const colmap::image_t image_id : recon.RegImageIds()) {
    config.AddImage(image_id);
  }
  if (no_inertial) {
    // no inertial factors -> fix full gauge incl. scale like standard BA
    config.FixGauge(colmap::BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);
  } else {
    // inertial factors carry scale + roll/pitch; fix remaining 6 dof by
    // holding the first keyframe pose constant
    config.SetConstantRigFromWorldPose(t_to_frame_id.begin()->second);
  }

  auto ba = colmap::CreateDefaultCeresBundleAdjuster(options, config, recon);
  ceres::Problem& problem = *ba->Problem();

  // ---- add inertial factors ----
  size_t added_rel = 0, added_rp = 0;
  auto make_factor_loss = [&]() -> ceres::LossFunction* {
    return factor_huber > 0 ? new ceres::HuberLoss(factor_huber) : nullptr;
  };
  if (!no_inertial) {
    for (const auto& f : rel_factors) {
      auto it_i = t_to_frame.find(f.t_i_ns);
      auto it_j = t_to_frame.find(f.t_j_ns);
      if (it_i == t_to_frame.end() || it_j == t_to_frame.end()) continue;
      double* pose_i = it_i->second->RigFromWorld().params.data();
      double* pose_j = it_j->second->RigFromWorld().params.data();
      if (!problem.HasParameterBlock(pose_i) ||
          !problem.HasParameterBlock(pose_j)) {
        continue;
      }
      problem.AddResidualBlock(BasaltRelPoseCostFunctor::Create(f, q_C, t_C),
                               make_factor_loss(), pose_i, pose_j);
      added_rel++;
    }
    for (const auto& f : rp_factors) {
      auto it = t_to_frame.find(f.t_ns);
      if (it == t_to_frame.end()) continue;
      double* pose = it->second->RigFromWorld().params.data();
      if (!problem.HasParameterBlock(pose)) continue;
      problem.AddResidualBlock(BasaltRollPitchCostFunctor::Create(f, q_C),
                               make_factor_loss(), pose);
      added_rp++;
    }
  }
  // ---- IMU preintegration factors (velocity + shared bias states) ----
  std::map<int64_t, std::array<double, 3>> velocities;
  std::array<double, 6> bias = imu_init_bias;
  size_t added_imu = 0;
  if (!imu_factors.empty()) {
    for (const auto& f : imu_factors) {
      for (const int64_t tk : {f.t_i_ns, f.t_j_ns}) {
        auto iv = imu_init_vel.find(tk);
        velocities.emplace(tk, iv != imu_init_vel.end()
                                   ? iv->second
                                   : std::array<double, 3>{0, 0, 0});
      }
    }
    for (const auto& f : imu_factors) {
      auto it_i = t_to_frame.find(f.t_i_ns);
      auto it_j = t_to_frame.find(f.t_j_ns);
      if (it_i == t_to_frame.end() || it_j == t_to_frame.end()) continue;
      double* pose_i = it_i->second->RigFromWorld().params.data();
      double* pose_j = it_j->second->RigFromWorld().params.data();
      if (!problem.HasParameterBlock(pose_i) ||
          !problem.HasParameterBlock(pose_j)) {
        continue;
      }
      problem.AddResidualBlock(
          ImuPreintCostFunctor::Create(f, q_C, t_C, gravity),
          make_factor_loss(), pose_i, pose_j,
          velocities[f.t_i_ns].data(), velocities[f.t_j_ns].data(),
          bias.data());
      added_imu++;
    }
  }
  LOG(INFO) << "Added " << added_rel << " rel-pose, " << added_rp
            << " roll-pitch and " << added_imu
            << " IMU-preintegration residuals to BA problem";

  ba->Solve();

  if (added_imu > 0) {
    double vmax = 0, vmed = 0;
    std::vector<double> sp;
    for (const auto& [t, v] : velocities) {
      const double n = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
      sp.push_back(n);
      vmax = std::max(vmax, n);
    }
    std::sort(sp.begin(), sp.end());
    if (!sp.empty()) vmed = sp[sp.size() / 2];
    LOG(INFO) << "IMU states: accel bias [" << bias[0] << ", " << bias[1]
              << ", " << bias[2] << "]  gyro bias [" << bias[3] << ", "
              << bias[4] << ", " << bias[5] << "]";
    LOG(INFO) << "Speeds: median " << vmed << " m/s, max " << vmax << " m/s";
  }

  recon.WriteText(output_path);
  LOG(INFO) << "Refined model written to " << output_path;
  return 0;
}
