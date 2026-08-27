#!/bin/bash
CM=/data/Hilti/colmap/floor_EG_run2
cd $CM || exit 1
declare -A SEEDS=(
 [OPENCV_FISHEYE]="465.3015482593691,465.32303798346413,730.0455886686005,720.1427007671206,0.0258007189,-0.0109092408,-0.0016899538,0.0001476680"
 [EUCM]="465.2979536302252,465.3194431883040,730.0455886686005,720.1427007671206,0.6899954350657926,0.8911981210457725"
 [FOV]="465.3015482593691,465.32303798346413,730.0455886686005,720.1427007671206,0.9"
 [THIN_PRISM_FISHEYE]="465.3015482593691,465.32303798346413,730.0455886686005,720.1427007671206,0.0258007189,-0.0109092408,-0.0016899538,0.0001476680,0,0,0,0"
)
for M in OPENCV_FISHEYE EUCM FOV THIN_PRISM_FISHEYE; do
  DB=db3_$M.db3; OUT=sparse3_$M
  echo "=== MODEL=$M"
  rm -f $DB; rm -rf $OUT; mkdir -p $OUT
  /build/src/colmap/exe/colmap feature_extractor --database_path $DB --image_path images --ImageReader.mask_path masks \
    --ImageReader.camera_model $M --ImageReader.single_camera_per_folder 1 \
    --ImageReader.camera_params "${SEEDS[$M]}" --FeatureExtraction.use_gpu 1 2>&1 | tail -1
  /build/src/colmap/exe/colmap sequential_matcher --database_path $DB --FeatureMatching.use_gpu 1 \
    --SequentialMatching.overlap 20 --SequentialMatching.loop_detection 1 2>&1 | tail -1
  /build/src/colmap/exe/colmap mapper --database_path $DB --image_path images --output_path $OUT \
    --Mapper.ba_refine_focal_length 1 --Mapper.ba_refine_principal_point 1 2>&1 | tail -1
  for m in $OUT/*/; do echo "-- $m"; /build/src/colmap/exe/colmap model_analyzer --path $m 2>&1 | grep -E "Registered images|Points:|reprojection"; done
done
echo MODEL_BAKEOFF_DONE
