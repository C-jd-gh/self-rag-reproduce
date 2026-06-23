popqa数据acc：54.11
python retrieval_lm/run_short_form.py \
    --model_name /openbayes/home/models/selfrag_llama2_7b \
    --input_file eval_data_official/popqa_longtail_w_gs.jsonl \
    --output_file eval_data_official/popqa_result.json \
    --mode adaptive_retrieval \
    --max_new_tokens 100 \
    --threshold 0.2 \
    --metric match \
    --ndocs 10 \
    --use_groundness \
    --use_utility \
    --device cuda