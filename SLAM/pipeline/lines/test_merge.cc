// Merge invariance test -- the two P0 merge defects.
//
// Two valid fragments of ONE physical edge, planes 0.2 deg apart, the second
// detected with REVERSED endpoint order (so its stored normal is negated --
// a line has no orientation, this is legal detector output).
//
// Before the fix: the group seed's normal was accumulated without sign
// alignment, so the reversed fragment cancelled it (measured 89.9 deg normal
// error), and the merged endpoints were built on the SEED's plane while the
// stored normal was the average -- the observation contradicted itself.
//
// Required after the fix, for BOTH input orders:
//   1. merged normal within 0.5 deg of the true plane
//   2. n . b1u == n . b2u == 0 (one self-consistent plane)
//   3. endpoint reversal changes nothing
#include <cmath>
#include <cstdio>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "LineExtractor.h"

using namespace ORB_SLAM3;
using V3 = Eigen::Vector3f;

static LineObs frag(const Eigen::Matrix3f& R, float th1, float th2, bool reversed) {
    // bearings at angles th on the unit circle of the z=0 plane, rotated by R:
    // R = I is the true plane, R = small tilt is the second fragment's plane.
    // One shared parameterisation, so the two arcs actually adjoin.
    V3 b1 = (R * V3(std::cos(th1), std::sin(th1), 0.f)).normalized();
    V3 b2 = (R * V3(std::cos(th2), std::sin(th2), 0.f)).normalized();
    if (reversed) std::swap(b1, b2);
    LineObs o;
    o.b1u = b1; o.b2u = b2;
    o.n = b1.cross(b2).normalized();     // what the detector actually stores
    o.dir = (b2 - b1).normalized();
    o.angLen = std::fabs(th2 - th1);
    o.cam = 0;
    return o;
}

int main() {
    const float deg = float(M_PI) / 180.f;
    const V3 nTrue(0.f, 0.f, 1.f);
    // second plane tilted 0.2 deg about x
    const Eigen::Matrix3f Rtilt =
        Eigen::AngleAxisf(0.2f * deg, V3::UnitX()).toRotationMatrix();

    LineExtractor ex;             // defaults: 0.5 deg normal tol, gap 1x
    ex.mpCamForMerge = nullptr;   // no pixel projection needed

    int fails = 0;
    for (int order = 0; order < 2; ++order) {
        std::vector<LineObs> in;
        LineObs A = frag(Eigen::Matrix3f::Identity(), 0.f * deg, 10.f * deg, false);
        LineObs B = frag(Rtilt, 11.f * deg, 21.f * deg, true);   // REVERSED
        if (order == 0) { in = {A, B}; } else { in = {B, A}; }

        auto out = ex.MergeGreatCircles(in);
        if (out.size() != 1) {
            std::printf("FAIL order %d: expected 1 merged obs, got %zu\n",
                        order, out.size());
            fails++; continue;
        }
        const LineObs& m = out[0];
        const float errN = std::acos(std::min(1.f, std::fabs(m.n.dot(nTrue)))) / deg;
        const float c1 = std::fabs(m.n.dot(m.b1u)), c2 = std::fabs(m.n.dot(m.b2u));
        std::printf("order %d: normal err %.3f deg | n.b1 %.2e n.b2 %.2e | arc %.2f deg\n",
                    order, errN, c1, c2, m.angLen / deg);
        if (errN > 0.5f) { std::printf("FAIL: normal error > 0.5 deg\n"); fails++; }
        if (c1 > 1e-5f || c2 > 1e-5f) {
            std::printf("FAIL: endpoints off their own plane\n"); fails++;
        }
    }
    std::printf(fails ? "MERGE TEST FAILED (%d)\n" : "MERGE TEST PASS\n", fails);
    return fails ? 1 : 0;
}
