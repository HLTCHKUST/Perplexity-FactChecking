TEST_DATA_PATH=/path/to/test/data.jsonl # e.g. covid_scientific.jsonl
EXP_NAME=output-name

LM_MODEL_TYPE=bert-base # bert-large
python main.py \
    --train_data_file=$TEST_DATA_PATH \
    --output_eval_file=/path-to-project/ppl_results/$LM_MODEL_TYPE.$EXP_NAME.npy \
    --model_name=$LM_MODEL_TYPE