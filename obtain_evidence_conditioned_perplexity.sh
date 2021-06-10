# covid myth
COVID_MYTH_PATH='data/covid_scientific.jsonl'
COVID_MYTH_EXP_NAME=covid_scientific

# covid politifact
COVID_POLITIFACT_W_JUSTIFICATION_PATH='data/covid_social.jsonl'
COVID_POLITIFACT_EXP_NAME=covid_politifact_justification

# fever
FEVER_TEST_PATH='data/fever_test.jsonl'
FEVER_TEXT_EXP_NAME=fever_test

PATHS=( $COVID_MYTH_PATH $COVID_POLITIFACT_W_JUSTIFICATION_PATH $FEVER_TEST_PATH )
NAMES=( $COVID_MYTH_EXP_NAME $COVID_POLITIFACT_EXP_NAME $FEVER_TEXT_EXP_NAME )

LM_MODEL_TYPE=gpt2 # Some options: gpt2 gpt2-xl gpt2 gpt2-medium gpt2-large gpt2-xl

mkdir -p ppl_results

for i in 0 #1 2
do
    INPUT_FILE_NAME=${PATHS[$i]}
    EXP_NAME=${NAMES[$i]}
    CUDA_VISIBLE_DEVICES=0 python run_language_modelling_clm.py \
        --model_name_or_path $LM_MODEL_TYPE \
        --data_file_path $INPUT_FILE_NAME \
        --do_eval \
        --per_gpu_train_batch_size 1 \
        --per_device_train_batch_size 1 \
        --block_size 128 \
        --per_gpu_eval_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --result_path ppl_results/$LM_MODEL_TYPE.$EXP_NAME.npy \
        --overwrite_output_dir 
done