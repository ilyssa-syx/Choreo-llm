## Text Annotation

### Segmentation
```bash
conda activate danceba
cd aist_annotation/segmentation
python segment.py --input_dir /network_space/server127_2/shared/caixhdata/aist_v/ --output_dir ./segment/test/ --filter_list ./utils/split/crossmodal_test.txt
```

detect beats:
```bash
cd aist_annotation/keyframe_detection
# please modify segment_dir, pickle_dir, output_dir
# pickle dir comes from EDGE/data/test/motion_sliced
python create_beat.py
```

Posescript annotation:
```bash
source /network_space/server127_2/shared/sunyx3/envs/posescript/bin/activate
cd aist_annotation/posescript/src/text2pose/generative_caption
# don't forget to rotate from smpl y-up to smpl z-up
python apply_aist_rotation.py
# AIST++ is originally 60 FPS, but keyframes comes in 30 FPS
python batch_custom_generative_caption.py --model_paths /network_space/server126/shared/sunyx/models/Choreo-llm/aist_annotation/posescript/experiments/capgen_CAtransfPSA2H2_dataPSA2ftPSH2/seed1/checkpoint_best.pth --pkl_dir ./motions_sliced_smpl_zup/test/ --json_dir /network_space/server126/shared/sunyx/models/Choreo-llm/aist_annotation/keyframe_detection/test --output_dir ./posescript_caption_annotation/test/ --fps_multiplier 2.0
```

Posefix annotation:
```bash
cd aist_annotation/posescript/src/text2pose/generative_modifier
ln -s /network_space/server126/shared/sunyx/models/Choreo-llm/aist_annotation/posescript/src/text2pose/generative_caption/motions_sliced_smpl_zup/ ./
python batch_custom_generative_modifier.py --model_paths /network_space/server126/shared/sunyx/models/Choreo-llm/aist_annotation/posescript/experiments/modgen_CAtransfPFAHPP_dataPFAftPFH/seed1/checkpoint_best.pth --pkl_dir ./motions_sliced_smpl_zup/test/ --json_dir /network_space/server126/shared/sunyx/models/Choreo-llm/aist_annotation/keyframe_detection/test --output_dir ./posefix_modifier_annotation/test/ --fps_multiplier 2.0
```

or to call the automated parser:
Posescript parser:
```bash
# remember that parser comes Y-up
cd /network_space/server126/shared/sunyx/models/Choreo-llm/aist_annotation/posescript/src/text2pose/posescript
conda activate pytorch3d
cp /network_space/storage43/sunyixuan/models/EDGE/data/test/motions_sliced/* ./motions_sliced_smpl_yup/test/
python custom_compute_coords.py
python custom_captioning.py
```

Posefix parser:
```bash
python custom_compute_rotation_change.py
python custom_correcting.py
```


call gemini:
```bash
cd aist_annotation/gemini_caption
# remember to modify caption and modifier directory
python interleave.py
python call_gemini.py --json_folder /network_space/server126/shared/sunyx/models/Choreo-llm/aist_annotation/keyframe_detection/test --video_folder /network_space/server127_2/shared/caixhdata/aist_v/ --output_folder ./gemini_caption/test/ --modifier_folder ./interleaved/test/ --prompt_file ./prompt.txt 
```

Merge:
```bash
python merge.py
```

## Choreography



Dance Model:

