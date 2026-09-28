// N consecutive frames through the REAL detect->merge->descriptor->LineTracker
// pipeline. The structural deliverable: edges must keep their identity through
// the clip -- same id, same physical edge, surviving brief misses.
//
// Output <out>_clip.csv: frame,trackId,age,x1,y1,x2,y2  (one row per matched
// or fresh observation; the viz colors by trackId). Plus a survival summary
// on stdout.
//
// dR = identity between frames (no IMU here): everything the tracker achieves
// in this tool it achieves on appearance + loose geometry alone; the SLAM
// additionally feeds it the gyro rotation.
//
// usage: track_clip <frames_dir> <first> <count> <mask> fx,..,k4 <out_prefix>
#include <cstdio>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

#include "ELSED.h"
#include "LineExtractor.h"

using namespace ORB_SLAM3;

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
    cv::Point2f project(const Eigen::Vector3f& b) const {
        const double r = std::sqrt(double(b.x())*b.x() + double(b.y())*b.y());
        const double th = std::atan2(r, (double)b.z());
        const double t2 = th*th;
        const double rd = th*(1 + k1*t2 + k2*t2*t2 + k3*t2*t2*t2 + k4*t2*t2*t2*t2);
        const double s = r > 1e-9 ? rd/r : 0.0;
        return cv::Point2f(float(fx*b.x()*s + cx), float(fy*b.y()*s + cy));
    }
};

static std::vector<LineObs> detect(const cv::Mat& img, const cv::Mat& mask,
                                   const KB4& cam, LineExtractor& ext) {
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
        lo.dir = (u2 - u1).normalized();
        lo.angLen = ang;
        lo.pxPerRad = ang > 1e-6f ? float(cv::norm(a - b)) / ang : 400.f;
        lo.cam = 0;
        LineExtractor::CanonicalizeObs(lo);
        out.push_back(lo);
    }
    out = ext.MergeGreatCircles(out);
    for (LineObs& o : out) {
        LineExtractor::CanonicalizeObs(o);
        std::vector<cv::Point2f> poly;
        const float th = std::acos(std::max(-1.f, std::min(1.f, o.b1u.dot(o.b2u))));
        for (int k = 0; k < 16; k++) {
            const float t = float(k) / 15.f;
            Eigen::Vector3f bk = th > 1e-6f
                ? ((std::sin((1-t)*th)*o.b1u + std::sin(t*th)*o.b2u) / std::sin(th)).normalized()
                : o.b1u;
            poly.push_back(cam.project(bk));
        }
        LineExtractor::ComputeBandDescriptor(img, poly, o);
    }
    return out;
}

int main(int argc, char** argv) {
    if (argc != 7) {
        std::fprintf(stderr, "usage: track_clip <frames_dir> <first> <count> "
                             "<mask> fx,..,k4 <out_prefix>\n");
        return 1;
    }
    const std::string dir = argv[1];
    const int f0 = atoi(argv[2]), nf = atoi(argv[3]);
    cv::Mat M = cv::imread(argv[4], cv::IMREAD_GRAYSCALE);
    KB4 cam{};
    sscanf(argv[5], "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf", &cam.fx, &cam.fy,
           &cam.cx, &cam.cy, &cam.k1, &cam.k2, &cam.k3, &cam.k4);
    const std::string out = argv[6];

    LineExtractor ext(1.5f, 40.0f, 30, 15);
    LineTracker trk;
    std::ofstream cf(out + "_clip.csv");
    cf << "frame,trackId,age,x1,y1,x2,y2\n";

    std::map<long, int> lifeOf;   // track id -> matched-frame count
    long nObsTotal = 0, nMatchedTotal = 0;
    for (int k = 0; k < nf; k++) {
        char name[512];
        snprintf(name, sizeof(name), "%s/%06d.jpg", dir.c_str(), f0 + k);
        cv::Mat img = cv::imread(name, cv::IMREAD_GRAYSCALE);
        if (img.empty()) { std::fprintf(stderr, "missing %s\n", name); return 2; }
        std::vector<LineObs> cur = detect(img, M, cam, ext);

        const size_t nBefore = trk.Tracks().size();
        std::vector<int> asg = trk.Match(cur, Eigen::Matrix3f::Identity());
        trk.Commit(cur, asg);

        // ids: matched obs carry mnLineId from Match; fresh tracks were
        // appended by Commit in cur order
        size_t freshAt = nBefore >= (size_t)trk.mStats.dropped
                       ? trk.Tracks().size() - trk.mStats.fresh : 0;
        size_t fresh = freshAt;
        for (size_t i = 0; i < cur.size(); i++) {
            long id; int age;
            if (asg[i] >= 0) { id = cur[i].mnLineId; age = -1;
                nMatchedTotal++;
                lifeOf[id]++;
            } else {
                id = trk.Tracks()[fresh].id; age = 1; fresh++;
                lifeOf[id] = 1;
            }
            nObsTotal++;
            cf << (f0 + k) << "," << id << "," << age << ","
               << cur[i].p1.x << "," << cur[i].p1.y << ","
               << cur[i].p2.x << "," << cur[i].p2.y << "\n";
        }
        if (k % 20 == 0)
            std::printf("frame %d: obs %zu matched %ld live tracks %zu\n",
                        f0 + k, cur.size(), trk.mStats.matched, trk.Tracks().size());
    }
    // survival: how long do identities live?
    std::vector<int> lives;
    for (auto& kv : lifeOf) lives.push_back(kv.second);
    std::sort(lives.begin(), lives.end());
    const size_t nT = lives.size();
    int n10 = 0; for (int v : lives) if (v >= 10) n10++;
    std::printf("tracks %zu | matched obs %ld/%ld (%.0f%%) | track life "
                "median %d p90 %d max %d | >=10 frames: %d\n",
                nT, nMatchedTotal, nObsTotal, 100.0 * nMatchedTotal / nObsTotal,
                nT ? lives[nT/2] : 0, nT ? lives[(size_t)(0.9*(nT-1))] : 0,
                nT ? lives.back() : 0, n10);
    return 0;
}
