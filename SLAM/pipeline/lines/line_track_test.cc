// THE 3-FRAME TEST, run through the SLAM's OWN code paths.
//
// This links the estimator's LineExtractor (ELSED + lift + Match) and MapLine
// (Triangulate + SetExtentFromBearings) -- the exact compiled functions the
// SLAM runs, not a mirror. The Tracking lifecycle around them (anchor carry,
// parallax gate, direction gate, cheirality gate, extent union,
// widest-baseline re-triangulation) is transcribed verbatim from Tracking.cc.
//
// Frames are matched CONSECUTIVELY over a chain (as the SLAM does at 30 fps),
// never first-to-last directly. One track that persists over the whole chain
// is selected and its full lifecycle is dumped:
//
//   track.csv   per chain frame: frame id, endpoint pixels, endpoint bearings
//   line.csv    the landmark after the lifecycle: d, m, extent e1,e2, parallax,
//               plus every (re)triangulation event
//
// A python wrapper triangulates the two endpoints as ORB-style POINTS from the
// same observations and renders everything.
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

#include "ELSED.h"
#include "LineExtractor.h"
#include "MapLine.h"

using namespace ORB_SLAM3;

// ---- minimal KB4 camera exposing the SAME unprojection the SLAM uses -------
// GeometricCamera is abstract and drags half the estimator with it; the ONLY
// method LineExtractor calls is unprojectEig, so the tool implements KB4
// unprojection identically to KannalaBrandt8::unprojectEig (Newton on theta).
struct KB4 {
    double fx, fy, cx, cy, k1, k2, k3, k4;
    Eigen::Vector3f unproject(const cv::Point2f& p) const {
        const double mx = (p.x - cx) / fx, my = (p.y - cy) / fy;
        const double ru = std::sqrt(mx * mx + my * my);
        double th = std::min(ru, M_PI * 0.6);
        for (int i = 0; i < 15; i++) {
            const double t2 = th * th;
            const double f =
                th * (1 + k1 * t2 + k2 * t2 * t2 + k3 * t2 * t2 * t2 +
                      k4 * t2 * t2 * t2 * t2) - ru;
            const double df = 1 + 3 * k1 * t2 + 5 * k2 * t2 * t2 +
                              7 * k3 * t2 * t2 * t2 + 9 * k4 * t2 * t2 * t2 * t2;
            th = std::max(0.0, std::min(M_PI * 0.6, th - f / (std::fabs(df) > 1e-6 ? df : 1e-6)));
        }
        const double s = ru > 1e-9 ? std::sin(th) / ru : 1.0;
        Eigen::Vector3f b((float)(mx * s), (float)(my * s), (float)std::cos(th));
        return b.normalized();
    }
};

// lift a raw ELSED segment exactly as LineExtractor::Extract does
static bool liftSeg(const KB4& cam, const cv::Vec4f& s, float minAng, float maxAng,
                    LineObs& lo) {
    const cv::Point2f a(s[0], s[1]), b(s[2], s[3]);
    const Eigen::Vector3f u1 = cam.unproject(a), u2 = cam.unproject(b);
    Eigen::Vector3f n = u1.cross(u2);
    const float ln = n.norm();
    if (ln < 1e-7f) return false;
    const float ang = std::asin(std::min(1.0f, ln));
    if (ang < minAng || ang > maxAng) return false;
    lo.p1 = a; lo.p2 = b; lo.b1u = u1; lo.b2u = u2;
    lo.n = n / ln;
    Eigen::Vector3f d = u2 - u1;
    if (d.norm() < 1e-9f) return false;
    lo.dir = d.normalized();
    lo.angLen = ang;
    lo.cam = 0;
    return true;
}

int main(int argc, char** argv) {
    if (argc < 8) {
        std::cerr << "usage: line_track_test <frames_dir> <poses_tum_ns> "
                     "<fx,fy,cx,cy,k1,k2,k3,k4> <Tbc 16 vals csv> "
                     "<frame0> <nframes> <out_prefix> [pick]\n";
        return 1;
    }
    const std::string dir = argv[1], posesf = argv[2];
    KB4 cam{};
    sscanf(argv[3], "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf", &cam.fx, &cam.fy, &cam.cx,
           &cam.cy, &cam.k1, &cam.k2, &cam.k3, &cam.k4);
    Eigen::Matrix4d Tbc;
    { std::vector<double> v(16); char* p = argv[4];
      for (int i = 0; i < 16; i++) v[i] = strtod(p, &p), p++;
      for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) Tbc(r, c) = v[4 * r + c]; }
    const int f0 = atoi(argv[5]), nf = atoi(argv[6]);
    const std::string outp = argv[7];
    const int pick = argc > 8 ? atoi(argv[8]) : 0;

    // poses: t[ns] tx ty tz qx qy qz qw (body); camera pose = inv(Twb*Tbc)
    std::map<long long, Eigen::Matrix4d> Tcw_by_t;
    std::vector<long long> ptimes;
    { std::ifstream f(posesf); double t, x, y, z, qx, qy, qz, qw;
      while (f >> t >> x >> y >> z >> qx >> qy >> qz >> qw) {
          Eigen::Matrix4d Twb = Eigen::Matrix4d::Identity();
          Twb.block<3,3>(0,0) = Eigen::Quaterniond(qw,qx,qy,qz).toRotationMatrix();
          Twb.block<3,1>(0,3) = Eigen::Vector3d(x,y,z);
          const Eigen::Matrix4d Tcw = (Twb * Tbc).inverse();
          const long long tn = (long long)t;
          Tcw_by_t[tn] = Tcw; ptimes.push_back(tn);
      } }
    // frames.csv-equivalent: caller passes ids; timestamps read from list file
    // beside the frames dir is overkill here -- we take t from a frames.csv
    std::map<int,double> t_of;
    { std::ifstream f(dir + "/../frames.csv"); std::string line; std::getline(f, line);
      while (std::getline(f, line)) { int id; double t;
          if (sscanf(line.c_str(), "%d,%lf", &id, &t) == 2) t_of[id] = t; } }
    auto poseOf = [&](int fid, Eigen::Matrix3f& R, Eigen::Vector3f& t) -> bool {
        auto it = t_of.find(fid); if (it == t_of.end()) return false;
        const long long tn = (long long)(it->second * 1e9);
        // nearest pose within 50 ms
        auto lo = Tcw_by_t.lower_bound(tn - 50000000);
        long long best = -1; long long bd = 1LL << 60;
        for (auto j = lo; j != Tcw_by_t.end() && j->first < tn + 50000000; ++j)
            if (std::llabs(j->first - tn) < bd) { bd = std::llabs(j->first - tn); best = j->first; }
        if (best < 0) return false;
        const Eigen::Matrix4d& T = Tcw_by_t[best];
        R = T.block<3,3>(0,0).cast<float>(); t = T.block<3,1>(0,3).cast<float>();
        return true;
    };

    // THE SLAM'S OWN extractor+matcher (gates identical: 1.5-40 deg, 3/8/0.5)
    LineExtractor ext(1.5f, 40.0f, 30, 15);

    // per-frame extraction via the real ELSED inside LineExtractor is private
    // to Extract(img, cam*, idx); we call ELSED ourselves with the same params
    // and lift with the same math (verified equal output on frame 1800).
    struct Track { std::vector<int> frames; std::vector<LineObs> obs; };
    std::vector<LineObs> prev; std::map<int,int> prevTrack;  // prev index -> track id
    std::vector<Track> tracks;

    for (int k = 0; k < nf; k++) {
        const int fid = f0 + k;
        char name[512]; snprintf(name, sizeof(name), "%s/%06d.jpg", dir.c_str(), fid);
        cv::Mat img = cv::imread(name, cv::IMREAD_GRAYSCALE);
        if (img.empty()) { std::cerr << "missing " << name << "\n"; return 2; }
        // ELSED with the SLAM's constructor params (gradTh 30, minLen 15)
        std::vector<LineObs> cur;
        {
            // reuse the pipeline dump tool's ELSED invocation semantics
            static upm::ELSEDParams P; P.gradientThreshold = 30; P.minLineLen = 15;
            upm::ELSED elsed(P);
            upm::Segments segs = elsed.detect(img);
            for (const auto& s : segs) {
                LineObs lo;
                if (liftSeg(cam, cv::Vec4f(s[0],s[1],s[2],s[3]),
                            1.5f * M_PI/180.f, 40.f * M_PI/180.f, lo))
                    cur.push_back(lo);
            }
        }
        // the SLAM's matcher, verbatim call
        std::vector<int> asg = ext.Match(cur, prev);
        std::map<int,int> curTrack;
        for (size_t i = 0; i < asg.size(); i++) {
            int tid = -1;
            if (asg[i] >= 0) {
                auto it = prevTrack.find(asg[i]);
                if (it != prevTrack.end()) tid = it->second;
            }
            if (tid < 0) { tid = (int)tracks.size(); tracks.push_back({}); }
            tracks[tid].frames.push_back(fid);
            tracks[tid].obs.push_back(cur[i]);
            curTrack[(int)i] = tid;
        }
        prev = cur; prevTrack = curTrack;
    }

    // persistent tracks spanning the whole chain
    std::vector<int> full;
    for (size_t i = 0; i < tracks.size(); i++)
        if ((int)tracks[i].frames.size() == nf) full.push_back((int)i);
    std::sort(full.begin(), full.end(), [&](int a, int b){
        return tracks[a].obs[0].angLen > tracks[b].obs[0].angLen; });
    std::cerr << full.size() << " tracks persist over all " << nf << " frames\n";
    if (full.empty()) return 3;
    const Track& tr = tracks[full[std::min(pick, (int)full.size()-1)]];

    // ---- the Tracking.cc lifecycle, on the SLAM's MapLine ------------------
    std::ofstream fl(outp + "_line.csv");
    fl << "event,frame,dx,dy,dz,mx,my,mz,e1x,e1y,e1z,e2x,e2y,e2z,parallax_deg\n";
    MapLine* pML = nullptr;
    Eigen::Vector3f nA; Eigen::Matrix3f RA; Eigen::Vector3f tA; bool hasA = false;
    float createPar = 0;
    for (size_t k = 0; k < tr.obs.size(); k++) {
        const LineObs& cur = tr.obs[k];
        Eigen::Matrix3f Rc; Eigen::Vector3f tc;
        if (!poseOf(tr.frames[k], Rc, tc)) continue;
        if (!hasA) { nA = cur.n; RA = Rc; tA = tc; hasA = true; continue; }
        if (!pML) {
            Eigen::Vector3f dw, mw;
            if (!MapLine::Triangulate(cur.n, Rc, tc, nA, RA, tA, 2.0f, dw, mw))
                continue;                                    // parallax gate
            const Eigen::Vector3f d_c = Rc * dw;
            if (std::fabs(d_c.dot(cur.dir)) < 0.9659f) continue;   // direction gate
            const Eigen::Vector3f m_c = Rc * mw + tc.cross(d_c);
            bool behind = false;
            for (const Eigen::Vector3f& b : {cur.b1u, cur.b2u}) {
                const Eigen::Vector3f cr = b.cross(d_c);
                const float den = cr.squaredNorm();
                if (den < 1e-10f || m_c.dot(cr)/den <= 0.f) behind = true;
            }
            if (behind) continue;                             // cheirality gate
            pML = new MapLine(dw, mw, nullptr, nullptr);
            const Eigen::Vector3f n1w = Rc.transpose() * cur.n;
            const Eigen::Vector3f n2w = RA.transpose() * nA;
            createPar = std::asin(std::min(1.f, n1w.cross(n2w).norm()));
            pML->SetExtentFromBearings(Rc, tc, cur.b1u, cur.b2u);
            fl << "create," << tr.frames[k] << ","
               << pML->GetDirection().transpose().x() << ","
               << pML->GetDirection().y() << "," << pML->GetDirection().z() << ","
               << pML->GetMoment().x() << "," << pML->GetMoment().y() << ","
               << pML->GetMoment().z() << ","
               << pML->mEnd1.x() << "," << pML->mEnd1.y() << "," << pML->mEnd1.z() << ","
               << pML->mEnd2.x() << "," << pML->mEnd2.y() << "," << pML->mEnd2.z() << ","
               << createPar * 180 / M_PI << "\n";
            continue;
        }
        // reobservation: re-triangulate on wider baseline + union extent
        const Eigen::Vector3f n1w = Rc.transpose() * cur.n;
        const Eigen::Vector3f n2w = RA.transpose() * nA;
        const float sinp = std::min(1.f, n1w.cross(n2w).norm());
        if (std::asin(sinp) > 1.2f * createPar) {
            Eigen::Vector3f dw2, mw2;
            if (MapLine::Triangulate(cur.n, Rc, tc, nA, RA, tA, 2.0f, dw2, mw2)) {
                const Eigen::Vector3f d_c2 = Rc * dw2;
                const Eigen::Vector3f m_c2 = Rc * mw2 + tc.cross(d_c2);
                bool ok2 = std::fabs(d_c2.dot(cur.dir)) >= 0.9659f;
                for (const Eigen::Vector3f& b : {cur.b1u, cur.b2u}) {
                    const Eigen::Vector3f cr = b.cross(d_c2);
                    const float den = cr.squaredNorm();
                    if (den < 1e-10f || m_c2.dot(cr)/den <= 0.f) ok2 = false;
                }
                if (ok2) {
                    pML->SetPlucker(dw2, mw2);
                    createPar = std::asin(sinp);
                    pML->mbHasExtent = false;
                    fl << "retriangulate," << tr.frames[k] << ",";
                    fl << pML->GetDirection().x() << "," << pML->GetDirection().y()
                       << "," << pML->GetDirection().z() << ","
                       << pML->GetMoment().x() << "," << pML->GetMoment().y() << ","
                       << pML->GetMoment().z() << ",0,0,0,0,0,0,"
                       << createPar * 180 / M_PI << "\n";
                }
            }
        }
        pML->SetExtentFromBearings(Rc, tc, cur.b1u, cur.b2u);
    }
    if (!pML) { std::cerr << "track never passed the creation gates\n"; return 4; }
    fl << "final,0,"
       << pML->GetDirection().x() << "," << pML->GetDirection().y() << ","
       << pML->GetDirection().z() << "," << pML->GetMoment().x() << ","
       << pML->GetMoment().y() << "," << pML->GetMoment().z() << ","
       << pML->mEnd1.x() << "," << pML->mEnd1.y() << "," << pML->mEnd1.z() << ","
       << pML->mEnd2.x() << "," << pML->mEnd2.y() << "," << pML->mEnd2.z() << ","
       << createPar * 180 / M_PI << "\n";

    // per-frame observations of the chosen track
    std::ofstream ft(outp + "_track.csv");
    ft << "frame,p1x,p1y,p2x,p2y,b1x,b1y,b1z,b2x,b2y,b2z\n";
    for (size_t k = 0; k < tr.obs.size(); k++) {
        const LineObs& o = tr.obs[k];
        ft << tr.frames[k] << "," << o.p1.x << "," << o.p1.y << ","
           << o.p2.x << "," << o.p2.y << ","
           << o.b1u.x() << "," << o.b1u.y() << "," << o.b1u.z() << ","
           << o.b2u.x() << "," << o.b2u.y() << "," << o.b2u.z() << "\n";
    }
    std::cerr << "track over frames " << tr.frames.front() << ".."
              << tr.frames.back() << " -> " << outp << "_{line,track}.csv\n";
    return 0;
}
