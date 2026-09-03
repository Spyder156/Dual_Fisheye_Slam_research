// Geometry tests for MapLine. A synthetic world line, two known camera poses:
// the residual must vanish on the true line and grow off it, and triangulation
// must recover the line -- and must REFUSE when the two views are degenerate.
#include <cmath>
#include <cstdio>
#include <Eigen/Geometry>
#include "MapLine.h"
using namespace ORB_SLAM3;
using Eigen::Vector3f; using Eigen::Matrix3f;

static int pass = 0, fail = 0;
static void chk(const char* n, double got, double want, double tol) {
    bool ok = std::fabs(got - want) <= tol;
    printf("%-46s got %12.6g want %12.6g  %s\n", n, got, want, ok ? "PASS" : "<<< FAIL");
    ok ? ++pass : ++fail;
}

int main() {
    // world line: through P0, direction D
    Vector3f P0(2.f, 1.f, 5.f), D(0.f, 1.f, 0.f);
    D.normalize();
    Vector3f M = P0.cross(D);
    MapLine L(D, M, nullptr, nullptr);

    chk("direction is unit", L.GetDirection().norm(), 1.0, 1e-6);
    chk("moment orthogonal to direction", L.GetDirection().dot(L.GetMoment()), 0.0, 1e-6);

    // camera 1 at origin, identity
    Matrix3f R1 = Matrix3f::Identity();
    Vector3f t1(0, 0, 0);
    // a bearing to a point ON the line must have zero angular error
    Vector3f Q = P0 + 3.0f * D;
    chk("residual on the line ~ 0", L.AngularError(R1, t1, Q.normalized()), 0.0, 1e-5);
    // a bearing well off the line must not
    Vector3f Off = Q + Vector3f(1.5f, 0, 0);
    bool bigger = L.AngularError(R1, t1, Off.normalized()) > 0.05;
    chk("residual off the line is large", bigger ? 1.0 : 0.0, 1.0, 0.0);

    // sliding ALONG the line must not change the residual (aperture invariance)
    Vector3f Q2 = P0 + 9.0f * D;
    chk("residual invariant along the line",
        L.AngularError(R1, t1, Q2.normalized()), 0.0, 1e-5);

    // ---- triangulation from two genuinely different viewpoints -------------
    Matrix3f R2 = Eigen::AngleAxisf(0.25f, Vector3f::UnitY()).toRotationMatrix();
    Vector3f C2(1.5f, 0.f, 0.f);
    Vector3f t2 = -R2 * C2;
    Vector3f n1 = L.NormalInCamera(R1, t1);
    Vector3f n2 = L.NormalInCamera(R2, t2);
    Vector3f dw, mw;
    bool ok = MapLine::Triangulate(n1, R1, t1, n2, R2, t2, 2.0f, dw, mw);
    chk("triangulation succeeds", ok ? 1.0 : 0.0, 1.0, 0.0);
    if (ok) {
        chk("recovered direction matches", std::fabs(dw.dot(D)), 1.0, 1e-3);
        MapLine L2(dw, mw, nullptr, nullptr);
        chk("recovered line contains the point",
            L2.AngularError(R1, t1, Q.normalized()), 0.0, 1e-3);
    }

    // ---- degenerate: same viewpoint, must REFUSE ---------------------------
    Vector3f dd, mm;
    bool deg = MapLine::Triangulate(n1, R1, t1, n1, R1, t1, 2.0f, dd, mm);
    chk("degenerate pair refused", deg ? 1.0 : 0.0, 0.0, 0.0);

    printf("\n%d passed, %d FAILED\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
