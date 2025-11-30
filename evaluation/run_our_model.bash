
# mistral
cd ~ && python -m twkim.climate_to_text_stage2.mtl_weatherqa.inference_simple_pretrained \
	--json-path kse/ClimateToText/data/WeatherQA/gemini_element_captions_2020.json \
	--image-root kse/ClimateToText/data/WeatherQA \
	--num-samples 3 \
	--max-new-tokens 128 --num-beams 1 \
	--no-repeat-ngram-size 4 \
	--repetition-penalty 1.1 \
	--base-model-path /home/agi592/models/mistral-7b


# baseline
python inference_simple.py \
  --json-path /home/agi592/kse/ClimateToText/data/WeatherQA/gemini_element_captions_2020.json \
  --image-root /home/agi592/kse/ClimateToText/data/WeatherQA \
  --num-samples 3 \
  --max-new-tokens 80 --num-beams 1 \
  --no-repeat-ngram-size 4 --repetition-penalty 1.1 \
  --prompt "Forecast summary in 2-3 sentences. Focus on hazards, region, and timing." \
  --checkpoint-dir /home/agi592/twkim/checkpoints/climate_to_text_stage2_wqa_summary_only_ep1_r32_stage1 \
  --base-model-path /home/agi592/models/mistral-7b \
  --stage1-encoder-ckpt /home/agi592/kse/ClimateToText/stage1_curriculum_runs/step1_all_types/stage1_vision_encoder_mae.pt \
  --stage1-classifier-ckpt /home/agi592/csh/ClimateToText/checkpoints/standalone_cls_efficientnet/standalone_classifier_efficientnet_b0_best.pt \
  --encoder-mode perceiver_patch_mae



# MTL
python inference_simple.py \
  --json-path /home/agi592/kse/ClimateToText/data/WeatherQA/gemini_element_captions_2020.json \
  --image-root /home/agi592/kse/ClimateToText/data/WeatherQA \
  --num-samples 3 \
  --max-new-tokens 80 --num-beams 1 \
  --no-repeat-ngram-size 4 --repetition-penalty 1.1 \
  --prompt "Forecast summary in 2-3 sentences. Focus on hazards, region, and timing." \
  --checkpoint-dir /home/agi592/twkim/checkpoints/stage2_mtl_weatherqa_ep1_ddp \
  --base-model-path /home/agi592/models/mistral-7b \
  --stage1-encoder-ckpt /home/agi592/kse/ClimateToText/stage1_curriculum_runs/step1_all_types/stage1_vision_encoder_mae.pt \
  --stage1-classifier-ckpt /home/agi592/csh/ClimateToText/checkpoints/standalone_cls_efficientnet/standalone_classifier_efficientnet_b0_best.pt \
  --encoder-mode perceiver_patch_mae


# Frontier
python inference_simple.py \
  --json-path /home/agi592/kse/ClimateToText/data/WeatherQA/gemini_element_captions_2020.json \
  --image-root /home/agi592/kse/ClimateToText/data/WeatherQA \
  --num-samples 5 \
  --max-new-tokens 80 --num-beams 1 \
  --no-repeat-ngram-size 4 --repetition-penalty 1.1 \
  --prompt "Forecast summary in 2-3 sentences. Focus on hazards, region, and timing." \
  --checkpoint-dir /home/agi592/twkim/checkpoints/stage2_summary_centric_ep2_wu_norm_fixed_512_frozen_clip_revised_revert \
  --base-model-path /home/agi592/models/mistral-7b \
  --stage1-encoder-ckpt /home/agi592/kse/ClimateToText/step1_mcon_thea_ttd_clip/stage1_vision_encoder_mae.pt \
  --stage1-classifier-ckpt /home/agi592/csh/ClimateToText/checkpoints/standalone_cls_efficientnet/standalone_classifier_efficientnet_b0_best.pt \
  --encoder-mode clip


python eval2/infer_all.py   --json-path /home/agi592/kse/ClimateToText/data/WeatherQA/gemini_element_captions_2020.json   --image-root /home/agi592/kse/ClimateToText/data/WeatherQA   --num-times 10   --max-new-tokens 80   --pretrained-max-new-tokens 128   --num-beams 1   --no-repeat-ngram-size 4   --repetition-penalty 1.1   --prompt "Forecast summary in 2-3 sentences. Focus on hazards, region, and timing."   --base-model-path /home/agi592/models/mistral-7b   --baseline-ckpt /home/agi592/twkim/checkpoints/climate_to_text_stage2_wqa_summary_only_ep1_r32_stage1   --mtl-ckpt /home/agi592/twkim/checkpoints/stage2_mtl_weatherqa_ep1_ddp   --frontier-ckpt /home/agi592/twkim/checkpoints/stage2_summary_centric_ep2_wu_norm_fixed_512_frozen_clip_revised_revert   --stage1-encoder-ckpt /home/agi592/kse/ClimateToText/stage1_curriculum_runs/step1_all_types/stage1_vision_encoder_mae.pt   --stage1-classifier-ckpt /home/agi592/csh/ClimateToText/checkpoints/standalone_cls_efficientnet/standalone_classifier_efficientnet_b0_best.pt   --frontier-stage1-encoder-ckpt /home/agi592/kse/ClimateToText/step1_mcon_thea_ttd_clip/stage1_vision_encoder_mae.pt   --baseline-encoder-mode perceiver_patch_mae   --mtl-encoder-mode perceiver_patch_mae   --frontier-encoder-mode clip   --save-path eval2/sample_fix.json