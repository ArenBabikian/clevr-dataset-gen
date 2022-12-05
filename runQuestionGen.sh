ROOT='question_generation'
TEST=""
SUFFIXES="rel obj-ex obj-cnt"
SUFFIXES="rel"
SUFFIXES="obj-ex obj-cnt"
for SUFFIX in ${SUFFIXES}
do
  python $ROOT/generate_questions.py \
    --input_scene_file clevr-distinct/scenes.json \
    --metadata_file $ROOT/metadata.json \
    --synonyms_json $ROOT/synonyms_distinct.json \
    --template_dir data/templates-${SUFFIX} \
    --scene_start_idx 0 \
    --num_scenes 1000 \
    --templates_per_image 1 \
    --instances_per_template -1 \
    --output_questions_file clevr-distinct/questions-${SUFFIX}${TEST}.json
done

# rel, num_scenes = 5?
# obj, num_scenes = 1000

# --input_scene_file archive/scenes/CLEVR_val_scenes.json \
# num objects : scene ids
# 3 : 14, 12, 6, 
# 4 : 7, 
# 5 : 0, 11, 10
# 6 : 4
# 7 : 13, 5, 3
# 8 : 9, 2
# 9 : 8, 
# 10 : 1
