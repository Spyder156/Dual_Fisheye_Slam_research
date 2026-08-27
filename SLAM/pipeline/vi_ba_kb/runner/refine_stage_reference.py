        bad.append(f"|b_a| {extra['accel_bias_norm']:.2f} m/s^2")
    return bad


def stage_vi_align(ep, traj, out, log, tag, metric=False):
    """Linear scale/gravity/bias solve. Seconds, and its residual predicts
    whether the expensive refinement can work at all.

    `metric` pins the scale to 1 for a front-end that already outputs metres --
    see C.METRIC_BRANCHES for why that matters. Gravity, velocities and the
    accel bias are still taken from the solve; only the scale is overridden,
    and the solved value is kept as `measured_scale` because a large deviation
    from 1 is a genuine warning that the VIO's own scale has drifted.
    """
    run(_dk(C.IMG_CORE,
                ["python", "/pipeline/lib/vi_align.py", ep, str(traj),
                 "--out-dir", str(out), "--tag", tag],
                mounts=[(out, out), (traj.parent, traj.parent)]),
        C.TIMEOUTS_S["vi_align"], log / f"vi_align_{tag}.log")
    al = json.loads((out / f"{tag}.json").read_text())
    if metric:
        al["measured_scale"] = al.get("scale")
        al["scale"] = 1.0
        (out / f"{tag}.json").write_text(json.dumps(al, indent=1))
    return al


def stage_refine(ep, model_in, align_json, out, log, tag,
                 raw_traj=None):
    """IMU preintegration inside the refinement BA -- the largest single gain
    in the pipeline (d982 84 -> 95, d99e 87 -> 97)."""
    d = out / f"{tag}_refined"
    d.mkdir(parents=True, exist_ok=True)
    factors = out / f"{tag}_imu_factors.txt"
    prepared = out / f"{tag}_model_in"

    # Drop points whose depth is not actually constrained -- seen from <2 deg
    # of parallax, or tracked over <4 images. On 6a7d0bc7 that is 13377 of
    # 22690 points (59%), and feeding them to the solver is what made the
    # refinement destructive: DROID scored 94.0 raw, 76.8 refined without this
    # step and 92.0 with it. Project1's spec lists it as a required stage of
    # the refine tail; it was simply never wired in here.
    filtered = out / f"{tag}_sparse_filt"
    try:
        run(_dk(C.IMG_CORE,
                ["python", "/pipeline/lib/filter_lowparallax.py",
                 str(model_in), str(filtered),
                 str(C.FILTER_MIN_ANGLE_DEG), str(C.FILTER_MIN_TRACK)],
                mounts=[(out, out), (model_in.parent, model_in.parent)]),
            C.TIMEOUTS_S["refine"], log / f"filter_{tag}.log")
        if (filtered / "points3D.txt").exists():
            model_in = filtered
    except (StageFailed, StageTimeout) as e:
        (log / f"filter_{tag}.log").write_text(
            f"low-parallax filter failed, refining unfiltered: {e}\n")
    # MECKA_MODEL_PREALIGNED: the trajectory this model was triangulated from
    # already went through apply_scale, so it is metric AND gravity-aligned.
    # Without this the factor writer scales and rotates it a SECOND time -- the
    # model reached the solver at 9.95 m against 30.57 m of IMU and the BA
    # destroyed the poses resolving the contradiction.
    run(_dk(C.IMG_CORE,
                ["python", "/pipeline/lib/make_imu_factors.py", ep,
                 str(model_in), str(align_json), str(prepared), str(factors)],
                env={"MECKA_MODEL_PREALIGNED": "1"},
                mounts=[(out, out), (model_in.parent, model_in.parent)]),
        C.TIMEOUTS_S["refine"], log / f"factors_{tag}.log")
    # Rel-pose factors tie the solution to the front-end's own local motion.
    # Without them the BA has nothing anchoring it to what was measured -- the
    # structure was triangulated from these very poses, so the visual residual
    # cannot say where the cameras really are, and the solver drifts. This is
    # the factor set Project1's delivered runs used and ours never had.
    # Roll-pitch: per-keyframe absolute tilt from CoreMotion. This is the only
    # ABSOLUTE constraint in the problem. Rel-pose and IMU preintegration both
    # tie neighbours, so a drift accumulating over hundreds of frames costs
    # almost nothing per pair -- the poses drifted 0.79 deg while the visual
    # evidence justified 0.023 deg. Adding this took the hard episode from 84
    # to 91, past the raw 88.8.
    rp = out / f"{tag}_rollpitch.txt"
    try:
        run(_dk(C.IMG_CORE,
                ["python", "/pipeline/lib/make_rollpitch_factors.py", ep,
                 str(prepared), str(rp)],
                mounts=[(out, out)]),
            C.TIMEOUTS_S["refine"], log / f"rollpitch_{tag}.log")
    except (StageFailed, StageTimeout) as e:
        (log / f"rollpitch_{tag}.log").write_text(
            f"roll-pitch factors unavailable: {e}\n")

    rel = out / f"{tag}_relpose.txt"
    factor_file = "/pipeline/configs/empty_factors.txt"
    try:
        args_rel = ["python", "/pipeline/lib/make_relpose_factors.py",
                    str(raw_traj), str(prepared), str(rel)]
        if rp.exists():
            args_rel += ["--rollpitch", str(rp)]
        run(_dk(C.IMG_CORE, args_rel, mounts=[(out, out)]),
            C.TIMEOUTS_S["refine"], log / f"relpose_{tag}.log")
        if rel.exists():
            factor_file = str(rel)
    except (StageFailed, StageTimeout) as e:
        (log / f"relpose_{tag}.log").write_text(
            f"rel-pose factors failed, refining without them: {e}\n")

    # Budget scaled to the video, and applied to the STALL timer as well as the
    # timeout: colmap_vi_ba is silent for the whole solve, so the stall timer is
    # what actually kills it. With the flat 1200 s stall a 75 s clip could hold
    # the box for 20 minutes with 199 episodes queued behind it.
    ba_dur = _traj_stats(raw_traj)["duration_s"] if raw_traj else None
    budget = C.ba_budget_s(ba_dur)
    print(f"    VI-BA budget {budget:.0f}s"
          + (f" (video {ba_dur:.0f}s)" if ba_dur else ""), flush=True)
    ba = run(_dk(C.IMG_BA,
                ["colmap_vi_ba", "--input_path", str(prepared),
                 "--output_path", str(d),
                 "--factors", factor_file,
                 "--imu_preint", str(factors),
                 "--factor_huber", str(C.BA_FACTOR_HUBER),
                 "--max_iterations", str(C.BA_MAX_ITERS)],
                mounts=[(out, out)]),
        budget, log / f"viba_{tag}.log", stall_s=budget)
    if "NO_CONVERGENCE" in ba:
        # not fatal -- the result is still usable and gets scored like any
        # other candidate -- but it is recorded so a fleet run can be asked
        # "which episodes did not settle" instead of that being invisible
        print(f"    NOTE: VI-BA did not converge within {C.BA_MAX_ITERS} "
              f"iterations on {tag}", flush=True)
    # --unrotate: the solver works in a gravity-aligned world, and the refined
    # model comes back in THAT frame. The trajectory it gets densified against
    # is still in the original one, so without undoing the rotation an internal
    # convention becomes a global attitude error in the shipped result. The
    # solver does not copy the sidecar, so point at the prepared model.
    tum = out / f"{tag}_refined_kf.tum"
    run(_dk(C.IMG_CORE,
                ["python", "/pipeline/lib/model_to_tum.py", str(d), str(tum),
                 "--unrotate", str(prepared / "world_transform.json")],
                mounts=[(out, out)]),
        C.TIMEOUTS_S["refine"], log / f"export_{tag}.log")

    # The BA only moves KEYFRAMES -- 386 of 3561 here. Shipping that is a
    # trajectory with holes and it looks exactly as sparse as it is.
    #
    # densify_trajectory (the one the Project1 deliverables used) propagates the
    # CORRECTION rather than the motion: at each optimised keyframe it takes
    # D_k = T_opt_k o inv(T_vio_k), interpolates D between keyframes (slerp on
    # rotation, lerp on translation) and applies it to the full-rate front-end
    # pose. Local front-end accuracy survives untouched; only global drift is
    # absorbed. That is why it densifies without distorting anything.
    dense = out / f"{tag}_refined.tum"
    try:
        run(_dk(C.IMG_CORE,
                ["python", "/pipeline/lib/densify_trajectory.py",
                 str(raw_traj), str(tum), str(dense)],
                mounts=[(out, out)]),
            C.TIMEOUTS_S["refine"], log / f"densify_{tag}.log")
        if dense.exists():
            return dense, d
    except (StageFailed, StageTimeout) as e:
        (log / f"densify_{tag}.log").write_text(
            f"densify failed, shipping keyframe-rate: {e}\n"
            f"{getattr(e, 'log', '')[-4000:]}\n")
    return tum, d


def stage_qa(ep, traj, out, log, tag, model=None):
    d = out / f"qa_{tag}"
    args = ["python", "/pipeline/lib/qa_episode.py", ep, str(traj), str(d)]
    if model:
        args.append(str(model))
    run(_dk(C.IMG_CORE, args,
                mounts=[(out, out), (traj.parent, traj.parent)]),
        C.TIMEOUTS_S["qa"], log / f"qa_{tag}.log")
    return json.loads((d / f"qa_{ep}.json").read_text())


# ---------------------------------------------------------------- branches
def branch_openvins(ep, ep_dir, out, log, calib, clock):
