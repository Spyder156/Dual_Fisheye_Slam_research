// TWO frames through the pipeline's OWN detect->mask->lift->merge->match, and
// nothing else. Links the compiled LineExtractor: MergeGreatCircles and Match
// are the estimator's functions, not mirrors. Output is three CSVs a python
// script draws:
//
//   <out>_segA.csv / _segB.csv   one row per observation AFTER mask+merge:
//       idx,x1,y1,x2,y2,b1x,b1y,b1z,b2x,b2y,b2z,angLen,pxPerRad
//       (pixels are the raw/merged ELSED ends; bearings are what the matcher
//        actually compares -- the viz projects bearings for merged arcs)
//   <out>_match.csv              iB,iA   (frame B obs matched to frame A obs,
//                                the same direction Tracking matches cur->prev)
//
// usage: two_frame_match <imgA> <imgB> <mask.png> fx,fy,cx,cy,k1,k2,k3,k4 <out>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

#include "ELSED.h"
#include "LineExtractor.h"

using namespace ORB_SLAM3;

// SAME spherical KB4 inverse as KannalaBrandt8::unprojectEig (Newton on theta)
struct KB4 {
    double fx, fy, cx, cy, k1, k2, k3, k4;
    Eigen::Vector3f unproject(const cv::Point2f& p) const {
        const double mx = (p.x - cx) / fx, my = (p.y - cy) / fy;
        const double ru = std::sqrt(mx * mx + my * my);
        double th = std::min(ru, M_PI * 0.6);
        for (int i = 0; i < 15; i++) {
            const double t2 = th * th;
            const double f = th * (1 + k1*t2 + k2*t2*t2 + k3*t2*t2*t2 + k4*t2*t2*t2*t2) - ru;
            const double df = 1 + 3*k1*t2 + 5*k2*t2*t2 + 7*k3*t2*t2*t2 + 9*k4*t2*t2*t2*t2;
            th = std::max(0.0, std::min(M_PI * 0.6, th - f / (std::fabs(df) > 1e-6 ? df : 1e-6)));
        }
        const double s = ru > 1e-9 ? std::sin(th) / ru : 1.0;
        Eigen::Vector3f b((float)(mx * s), (float)(my * s), (float)std::cos(th));
        return b.normalized();
    }
};

// detect + mask + lift, exactly as LineExtractor::Extract (ELSED params, mask
// on EITHER endpoint, angular-length window, pxPerRad from the segment's own
// pixel/angular length)
static std::vector<LineObs> detect(const cv::Mat& img, const cv::Mat& mask,
                                   const KB4& cam) {
    static upm::ELSEDParams P;
    P.gradientThreshold = 30; P.minLineLen = 15;
    upm::ELSED elsed(P);
    upm::Segments segs = elsed.detect(img);
    const float minAng = 1.5f * float(M_PI) / 180.f, maxAng = 40.f * float(M_PI) / 180.f;
    auto masked = [&](const cv::Point2f& p) {
        if (mask.empty()) return false;
        const int x = (int)std::round(p.x), y = (int)std::round(p.y);
        if (x < 0 || y < 0 || x >= mask.cols || y >= mask.rows) return true;
        return mask.at<uchar>(y, x) != 0;
    };
    std::vector<LineObs> out;
    for (const auto& s : segs) {
        const cv::Point2f a(s[0], s[1]), b(s[2], s[3]);
        if (masked(a) || masked(b)) continue;
        const Eigen::Vector3f u1 = cam.unproject(a), u2 = cam.unproject(b);
        Eigen::Vector3f n = u1.cross(u2);
        const float ln = n.norm();
        if (ln < 1e-7f) continue;
        const float ang = std::asin(std::min(1.0f, ln));
        if (ang < minAng || ang > maxAng) continue;
        LineObs lo;
        lo.p1 = a; lo.p2 = b; lo.b1u = u1; lo.b2u = u2;
        lo.n = n / ln;
        Eigen::Vector3f d = u2 - u1;
        if (d.norm() < 1e-9f) continue;
        lo.dir = d.normalized();
        lo.angLen = ang;
        lo.pxPerRad = ang > 1e-6f ? float(cv::norm(a - b)) / ang : 400.f;
        lo.cam = 0;
        out.push_back(lo);
    }
    return out;
}

static void dump(const std::string& path, const std::vector<LineObs>& v) {
    std::ofstream f(path);
    f << "idx,x1,y1,x2,y2,b1x,b1y,b1z,b2x,b2y,b2z,angLen,pxPerRad\n";
    for (size_t i = 0; i < v.size(); i++) {
        const LineObs& o = v[i];
        f << i << "," << o.p1.x << "," << o.p1.y << "," << o.p2.x << "," << o.p2.y
          << "," << o.b1u.x() << "," << o.b1u.y() << "," << o.b1u.z()
          << "," << o.b2u.x() << "," << o.b2u.y() << "," << o.b2u.z()
          << "," << o.angLen << "," << o.pxPerRad << "\n";
    }
}

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "usage: two_frame_match <imgA> <imgB> <mask> "
                     "fx,fy,cx,cy,k1,k2,k3,k4 <out_prefix>\n";
        return 1;
    }
    cv::Mat A = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    cv::Mat B = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
    cv::Mat M = cv::imread(argv[3], cv::IMREAD_GRAYSCALE);
    if (A.empty() || B.empty()) { std::cerr << "missing image\n"; return 2; }
    KB4 cam{};
    sscanf(argv[4], "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf", &cam.fx, &cam.fy,
           &cam.cx, &cam.cy, &cam.k1, &cam.k2, &cam.k3, &cam.k4);
    const std::string out = argv[5];

    LineExtractor ext(1.5f, 40.0f, 30, 15);   // the SLAM's construction
    // ASSUMPTION: mpCamForMerge stays null here, so a MERGED observation keeps
    // pxPerRad = 400 instead of its true local scale; raw observations carry
    // their own. Same in spirit, slightly different d_orth gate for merged obs.
    std::vector<LineObs> a = detect(A, M, cam);
    std::vector<LineObs> b = detect(B, M, cam);
    printf("detected (post-mask): A %zu  B %zu\n", a.size(), b.size());
    a = ext.MergeGreatCircles(a);             // the estimator's merge
    b = ext.MergeGreatCircles(b);
    printf("after merge:          A %zu  B %zu\n", a.size(), b.size());

    std::vector<int> asg = ext.Match(b, a);   // the estimator's matcher, cur=B prev=A
    int nm = 0;
    std::ofstream mf(out + "_match.csv");
    mf << "iB,iA\n";
    for (size_t i = 0; i < asg.size(); i++)
        if (asg[i] >= 0) { mf << i << "," << asg[i] << "\n"; nm++; }
    printf("matched: %d of B=%zu against A=%zu\n", nm, b.size(), a.size());

    dump(out + "_segA.csv", a);
    dump(out + "_segB.csv", b);
    return 0;
}
