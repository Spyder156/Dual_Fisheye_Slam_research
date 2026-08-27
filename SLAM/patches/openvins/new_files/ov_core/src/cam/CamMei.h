/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * (project-local extension: Mei/UCM + radtan camera model)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

#ifndef OV_CORE_CAM_MEI_H
#define OV_CORE_CAM_MEI_H

#include "CamBase.h"

namespace ov_core {

/**
 * @brief Mei / unified camera model (UCM) with radtan distortion ("omni-radtan" in Kalibr).
 *
 * Projection: a unit-sphere point Xs is projected through a pinhole displaced by xi:
 *   x = Xs_x / (Xs_z + xi),  y = Xs_y / (Xs_z + xi)
 * followed by radtan distortion (k1, k2, p1, p2) and the pinhole K.
 *
 * The mirror parameter xi is treated as a FIXED constant (from factory / offline
 * calibration) so the online-calibrated intrinsic vector keeps the standard 8-dim
 * layout [fx fy cx cy k1 k2 p1 p2] and no state-size changes are needed.
 *
 * Input/output "normalized coordinates" follow the OpenVINS convention of
 * pinhole-normalized coordinates zn = (X/Z, Y/Z); rays with Z <= 0 (FOV beyond
 * 90 deg from the optical axis) are NOT representable in that convention and must
 * be excluded by a mask upstream.
 */
class CamMei : public CamBase {

public:
  /**
   * @param width Width of the camera (raw pixels)
   * @param height Height of the camera (raw pixels)
   * @param xi Mirror parameter (fixed, not estimated online)
   */
  CamMei(int width, int height, double xi) : CamBase(width, height), _xi(xi) {}

  ~CamMei() {}

  Eigen::Vector2f undistort_f(const Eigen::Vector2f &uv_dist) override {

    Eigen::MatrixXd cam_d = camera_values;

    // Remove K
    double xd = (uv_dist(0) - cam_d(2)) / cam_d(0);
    double yd = (uv_dist(1) - cam_d(3)) / cam_d(1);

    // Iteratively undo radtan distortion (fixed-point, same scheme as OpenCV)
    double x = xd, y = yd;
    for (int i = 0; i < 12; i++) {
      double r2 = x * x + y * y;
      double rad = 1.0 + cam_d(4) * r2 + cam_d(5) * r2 * r2;
      double dx = 2.0 * cam_d(6) * x * y + cam_d(7) * (r2 + 2.0 * x * x);
      double dy = cam_d(6) * (r2 + 2.0 * y * y) + 2.0 * cam_d(7) * x * y;
      x = (xd - dx) / rad;
      y = (yd - dy) / rad;
    }

    // UCM unprojection to the unit sphere, then to pinhole-normalized coords
    double r2 = x * x + y * y;
    double disc = 1.0 + (1.0 - _xi * _xi) * r2;
    if (disc < 0)
      disc = 0;
    double factor = (_xi + std::sqrt(disc)) / (1.0 + r2);
    double zs = factor - _xi;

    Eigen::Vector2f pt_out;
    if (std::abs(zs) < 1e-8) {
      // ray at exactly 90deg: not representable, return something finite
      pt_out(0) = (float)(factor * x * 1e8);
      pt_out(1) = (float)(factor * y * 1e8);
      return pt_out;
    }
    pt_out(0) = (float)(factor * x / zs);
    pt_out(1) = (float)(factor * y / zs);
    return pt_out;
  }

  Eigen::Vector2f distort_f(const Eigen::Vector2f &uv_norm) override {

    Eigen::MatrixXd cam_d = camera_values;

    // Pinhole-normalized -> UCM plane: x = a/(1 + xi*n), n = ||(a,b,1)||
    double a = uv_norm(0), b = uv_norm(1);
    double n = std::sqrt(a * a + b * b + 1.0);
    double denom = 1.0 + _xi * n;
    double x = a / denom, y = b / denom;

    // Radtan distortion
    double r2 = x * x + y * y;
    double rad = 1.0 + cam_d(4) * r2 + cam_d(5) * r2 * r2;
    double x1 = x * rad + 2.0 * cam_d(6) * x * y + cam_d(7) * (r2 + 2.0 * x * x);
    double y1 = y * rad + cam_d(6) * (r2 + 2.0 * y * y) + 2.0 * cam_d(7) * x * y;

    Eigen::Vector2f uv_dist;
    uv_dist(0) = (float)(cam_d(0) * x1 + cam_d(2));
    uv_dist(1) = (float)(cam_d(1) * y1 + cam_d(3));
    return uv_dist;
  }

  void compute_distort_jacobian(const Eigen::Vector2d &uv_norm, Eigen::MatrixXd &H_dz_dzn, Eigen::MatrixXd &H_dz_dzeta) override {

    Eigen::MatrixXd cam_d = camera_values;

    double a = uv_norm(0), b = uv_norm(1);
    double n = std::sqrt(a * a + b * b + 1.0);
    double denom = 1.0 + _xi * n;
    double x = a / denom, y = b / denom;

    // d(x,y)/d(a,b) for the UCM front-end
    double inv_n = 1.0 / n;
    double d2 = denom * denom;
    Eigen::Matrix2d dxy_dab;
    dxy_dab(0, 0) = 1.0 / denom - _xi * a * a * inv_n / d2;
    dxy_dab(0, 1) = -_xi * a * b * inv_n / d2;
    dxy_dab(1, 0) = -_xi * a * b * inv_n / d2;
    dxy_dab(1, 1) = 1.0 / denom - _xi * b * b * inv_n / d2;

    // Radtan jacobian wrt (x,y), same structure as CamRadtan
    double r2 = x * x + y * y;
    double rad = 1.0 + cam_d(4) * r2 + cam_d(5) * r2 * r2;
    double drad_dr2 = cam_d(4) + 2.0 * cam_d(5) * r2;
    Eigen::Matrix2d dxd_dxy;
    dxd_dxy(0, 0) = rad + 2.0 * x * x * drad_dr2 + 2.0 * cam_d(6) * y + 6.0 * cam_d(7) * x;
    dxd_dxy(0, 1) = 2.0 * x * y * drad_dr2 + 2.0 * cam_d(6) * x + 2.0 * cam_d(7) * y;
    dxd_dxy(1, 0) = 2.0 * x * y * drad_dr2 + 2.0 * cam_d(6) * x + 2.0 * cam_d(7) * y;
    dxd_dxy(1, 1) = rad + 2.0 * y * y * drad_dr2 + 6.0 * cam_d(6) * y + 2.0 * cam_d(7) * x;

    Eigen::Matrix2d K;
    K << cam_d(0), 0, 0, cam_d(1);

    H_dz_dzn = Eigen::MatrixXd::Zero(2, 2);
    H_dz_dzn = K * dxd_dxy * dxy_dab;

    // Distorted (pre-K) coordinates for the intrinsic jacobian
    double x1 = x * rad + 2.0 * cam_d(6) * x * y + cam_d(7) * (r2 + 2.0 * x * x);
    double y1 = y * rad + cam_d(6) * (r2 + 2.0 * y * y) + 2.0 * cam_d(7) * x * y;

    H_dz_dzeta = Eigen::MatrixXd::Zero(2, 8);
    H_dz_dzeta(0, 0) = x1;
    H_dz_dzeta(0, 2) = 1;
    H_dz_dzeta(0, 4) = cam_d(0) * x * r2;
    H_dz_dzeta(0, 5) = cam_d(0) * x * r2 * r2;
    H_dz_dzeta(0, 6) = 2.0 * cam_d(0) * x * y;
    H_dz_dzeta(0, 7) = cam_d(0) * (r2 + 2.0 * x * x);
    H_dz_dzeta(1, 1) = y1;
    H_dz_dzeta(1, 3) = 1;
    H_dz_dzeta(1, 4) = cam_d(1) * y * r2;
    H_dz_dzeta(1, 5) = cam_d(1) * y * r2 * r2;
    H_dz_dzeta(1, 6) = cam_d(1) * (r2 + 2.0 * y * y);
    H_dz_dzeta(1, 7) = 2.0 * cam_d(1) * x * y;
  }

  /// Fixed mirror parameter
  double get_xi() const { return _xi; }

protected:
  double _xi;
};

} // namespace ov_core

#endif /* OV_CORE_CAM_MEI_H */
