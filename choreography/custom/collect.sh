
cd /network_space/server126/shared/sunyx/models/Choreo-llm/aist_annotation/segmentation
python segment.py --input_dir --output_dir /network_space/server126/shared/sunyx/models/Choreo-llm/choreography/custom/data/segment

cd ../keyframe_detection
python create_beat.py 

cd ../gemini_caption
python call_gemini.py --no_modifier --prompt_file ./prompt.txt

