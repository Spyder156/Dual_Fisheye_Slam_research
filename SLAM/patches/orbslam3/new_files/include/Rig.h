/**
 * Rig.h -- non-overlapping multi-camera rig for ORB-SLAM3.
 *
 * ORB-SLAM3 ships two-camera plumbing (mpCamera2 / mTlr / Stereo.T_c1_c2) but it
 * is gated behind the STEREO sensor types and assumes the two views OVERLAP.
 * Our rig is back-to-back: the cameras share no field of view at any instant.
 * This module holds the rig as a first-class object so the rest of the system can
 * reason about it without inheriting the stereo assumptions.
 *
 * Extrinsics come from Step 0 (COLMAP rig-constrained BA + IMU metric scale),
 * NOT from an assumed 180 deg / zero baseline.
 *
 * Refinement policy (measured, not guessed): with non-overlapping cameras the
 * rig ROTATION is well observed but the TRANSLATION is not -- in Step 0, BA
 * improved the angle and simultaneously degraded the baseline by 24%. So the
 * default is: refine rotation, hold translation.
 */
#ifndef RIG_H
#define RIG_H

#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <sophus/se3.hpp>

namespace ORB_SLAM3 {

class Rig {
public:
    Rig() = default;

    /// Load from the settings yaml. Keys (all optional; absent => rig disabled):
    ///   Rig.enabled            : 1/0
    ///   Rig.T_c0_c1            : 4x4, camera1 -> camera0
    ///   Rig.refine_rotation    : 1/0  (default 1)
    ///   Rig.refine_translation : 1/0  (default 0, see header note)
    bool LoadFromSettings(const std::string& settingsPath);

    bool IsEnabled() const { return mbEnabled; }
    int NumCameras() const { return mbEnabled ? 2 : 1; }

    /// camera1 -> camera0 (cam0 is the rig reference sensor)
    const Sophus::SE3f& T_c0_c1() const { return mT_c0_c1; }
    Sophus::SE3f T_c1_c0() const { return mT_c0_c1.inverse(); }

    /// Angle between the two optical axes, degrees. 180 == ideal back-to-back.
    float InterCameraAngleDeg() const;
    /// Distance between the two optical centres, metres.
    float BaselineMetres() const;

    bool RefineRotation() const { return mbRefineRotation; }
    bool RefineTranslation() const { return mbRefineTranslation; }

    /// One-line summary for the log. Values are deliberately NOT printed by the
    /// stage hooks -- only by this, once, at load.
    void PrintSummary() const;

    /// Stage markers. These exist so the rig's path through the system is
    /// traceable in a log without dumping numbers on every frame.
    static void Stage(const char* where, const char* what);

private:
    bool mbEnabled = false;
    bool mbRefineRotation = true;
    bool mbRefineTranslation = false;
    Sophus::SE3f mT_c0_c1;
};

}  // namespace ORB_SLAM3

#endif
