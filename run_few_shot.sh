# ===============================================================================================================
#   EXP 1: COVID SCIENTIFIC
# ===============================================================================================================
COVID_SCIENTIFIC_PATH=/data/covid_scientific.jsonl
EXP_NAME=covid_scientific

for K in 50 10 2
do
    for LM_MODEL_TYPE in gpt2 #gpt2-medium gpt2-large gpt2-xl bert-base
    do
        python few_shot_ppl.py \
            --test_data_file $COVID_SCIENTIFIC_PATH \
            --test_result_path /ppl_results/$LM_MODEL_TYPE.$EXP_NAME.npy \
            --k $K \
            --covid_data \
            --exp_name $EXP_NAME
    done
done

# ===============================================================================================================
#   EXP 2: COVID SOCIAL
# ===============================================================================================================
COVID_POLITIFACT_PATH=/data/covid_social.jsonl
EXP_NAME=covid_politifact_justification

for K in 50 10 2
do
    for LM_MODEL_TYPE in gpt2 #gpt2-medium gpt2-large gpt2-xl bert-base
    do
        python few_shot_ppl.py \
            --test_data_file $COVID_POLITIFACT_PATH \
            --test_result_path /ppl_results/$LM_MODEL_TYPE.$EXP_NAME.npy \
            --k $K \
            --covid_data \
            --exp_name $EXP_NAME
    done
done

# ===============================================================================================================
#   EXP 3: FEVER
# ===============================================================================================================
FEVER_TRAIN_PATH=/data/fever_train.jsonl
TRAIN_EXP_NAME=fever_train_small

FEVER_TEST_PATH=/data/fever_test.jsonl
TEST_EXP_NAME=fever_test

EXP_NAME_FOR_SAVE=fever

for K in 50 10 2
do
    for LM_MODEL_TYPE in gpt2 #gpt2-medium gpt2-large gpt2-xl bert-base
    do
        python few_shot_ppl.py \
            --train_data_file $FEVER_TRAIN_PATH \
            --train_result_path /ppl_results/$LM_MODEL_TYPE.$TRAIN_EXP_NAME.npy \
            --test_data_file $FEVER_TEST_PATH \
            --test_result_path /ppl_results/$LM_MODEL_TYPE.$TEST_EXP_NAME.npy \
            --k $K \
            --exp_name $EXP_NAME_FOR_SAVE
    done
done

