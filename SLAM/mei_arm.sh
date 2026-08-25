#!/bin/bash
CM=/data/Hilti/colmap/floor_EG_run2
cd $CM || exit 1
CB=/build/src/colmap/exe/colmap
# xi, fx, fy from our KB4->Mei fit; k1,k2,p1,p2 from the same fit
SEED="1196.635,1196.635,730.0455886686005,720.1427007671206,1.57131,-0.01749,-0.33453,0.0,0.0"
DB=db3_MEI.db3; OUT=sparse3_MEI
rm -f $DB; rm -rf $OUT; mkdir -p $OUT
$CB feature_extractor --database_path $DB --image_path images --ImageReader.mask_path masks \
  --ImageReader.camera_model MEI --ImageReader.single_camera_per_folder 1 \
  --ImageReader.camera_params "$SEED" --FeatureExtraction.use_gpu 1 2>&1 | tail -2
$CB sequential_matcher --database_path $DB --FeatureMatching.use_gpu 1 \
  --SequentialMatching.overlap 20 --SequentialMatching.loop_detection 1 2>&1 | tail -1
$CB mapper --database_path $DB --image_path images --output_path $OUT \
  --Mapper.ba_refine_focal_length 1 --Mapper.ba_refine_principal_point 1 2>&1 | tail -1
for m in $OUT/*/; do echo "-- $m"; $CB model_analyzer --path $m 2>&1 | grep -E "Registered images|Points:|reprojection"; done
echo MEI_DONE
