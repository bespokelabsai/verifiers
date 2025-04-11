import verifiers as vf
from verifiers.tools import calculator, mean
from verifiers.prompts import CALCULATOR_FEW_SHOT
import time
import os 

os.environ["CURATOR_VIEWER"] = "1"

# from unsloth import FastLanguageModel
# model_name = "Qwen/Qwen2.5-7B-Instruct"
model_name = "Qwen/Qwen2.5-3B-Instruct"
# model_name = "/root/richard/test/verifiers/outputs/bfcl-qwen2.5-7b-instruct-1-turns-dr-grpo-update-ref-model-no-format-score-new-prompt-with-gibberish-judge/checkpoint-700"
model, tokenizer = vf.get_model_and_tokenizer(model_name)

NUM_GPUS = 4
PER_DEVICE_BATCH_SIZE = 8
NUM_GENERATIONS = 8
# (NUM_GPUS - 1) * PER_DEVICE_BATCH_SIZE / NUM_GENERATIONS = NUM PROMPT PER DEVICE
EVAL_DATASET_SIZE = 100
MAX_STEPS_PER_TURN = 10
CURRICULUM_LEARNING = True
MAX_TURNS_ONLY = True
MAX_NUM_TURNS = 1 # 4 turns is almost max
DEBUG_GENERATE = False
DEBUG_REWARDS = False
EVAL_STEPS = 50
SAVE_STEPS = 100
NUM_EPOCHS = 100
USE_DR_GRPO = False
USE_LATEST_TRL = False
UPDATE_REF_MODEL = True
EVAL_ON_START = False
TEST_HYPOTHESIS = False
if TEST_HYPOTHESIS:
    BASELINE_RUN = False
BETA = 0.001
MAX_GRAD_NORM = 0.2
NUM_ITERATIONS = 2
APPLY_OVERLONG_FILTERING = True

# if DEBUG_GENERATE or DEBUG_REWARDS:
#     NUM_GPUS = 2
#     MAX_STEPS_PER_TURN = 5

# Initialize tool environment for GSM8K
vf_env = vf.BfclEnv(
    dataset="bfcl",
    # few_shot=CALCULATOR_FEW_SHOT[0],
    tools=[],
    max_num_turns=MAX_NUM_TURNS,
    max_steps_per_turn=MAX_STEPS_PER_TURN,
    curriculum_learning=CURRICULUM_LEARNING,
    use_latest_trl=USE_LATEST_TRL,
)

train_dataset = vf_env.get_dataset(max_num_turns=MAX_NUM_TURNS)
# if TEST_HYPOTHESIS:
#     question_subset_ids = ["multi_turn_base_" + str(i) for i in [69, 198, 1, 134, 194, 93, 6, 104, 145, 138, 163, 47]]
#     train_dataset = train_dataset.filter(lambda x: x["id"] in question_subset_ids)
#     print(set(train_dataset["id"]))
#     assert len(set(train_dataset["id"])) == len(question_subset_ids), f"len(set(train_dataset['id'])): {len(set(train_dataset['id']))}, len(question_subset_ids): {len(question_subset_ids)}"
eval_dataset = vf_env.get_eval_dataset(max_num_turns=MAX_NUM_TURNS, max_turn_only=MAX_TURNS_ONLY, 
                                    #    n=EVAL_DATASET_SIZE
                                       )
# if TEST_HYPOTHESIS:
#     eval_dataset = eval_dataset.select(range(0,10))
# train_dataset = train_dataset.select(range(0,10))
# eval_dataset = eval_dataset.select(range(0,20))
print(train_dataset)
print(eval_dataset)
# raise Exception("Stop")
# time.sleep(5)

rubric = vf_env.get_rubric()

# notable defaults: lr = 1e-6, max_grad_norm = 0.01, constant lr 10 warmup steps, 1024 tokens in+out
run_name = "bfcl-" + model_name.split("/")[-1].lower() + f"-{MAX_NUM_TURNS}-turns"
if USE_LATEST_TRL:
    run_name += "-latest-trl"
if USE_DR_GRPO:
    run_name += "-dr-grpo"
if UPDATE_REF_MODEL:
    run_name += "-update-ref-model"

run_name += "-no-format-score"
run_name += "-new-prompt"
if TEST_HYPOTHESIS:
    run_name += "-test-hypothesis"
    if BASELINE_RUN:
        run_name += "-baseline"
    else:
        # run_name += "-clip-advantage-negative-only"
        run_name += "-clip-advantage-positive-only"
if APPLY_OVERLONG_FILTERING:
    run_name += "-apply-overlong-filtering"


    
# run_name += "-with-gibberish-judge"
# if DEBUG_GENERATE or DEBUG_REWARDS:
#     run_name += "-debug"
# run_name = "bfcl-3B-test-run"
training_args = vf.get_default_grpo_config(
    run_name=run_name,
    num_gpus=NUM_GPUS
)
training_args.num_train_epochs = NUM_EPOCHS
# rollouts per prompt
training_args.num_generations = NUM_GENERATIONS
if DEBUG_GENERATE or DEBUG_REWARDS:
    # training_args.num_generations = 2
    training_args.report_to = "none"
# minibatch size per GPU ( bs 6 * 7 gpus / 7 rollouts -> 6 prompts per batch)
training_args.per_device_train_batch_size = PER_DEVICE_BATCH_SIZE
# batches to accumulate (6 prompts * 4 -> 24 prompts per global batch)
training_args.gradient_accumulation_steps = 4
# steps per global batch (1 on-policy, 1 off-policy)
# NOTE: In TRL 0.15.2 this is not supported
training_args.num_iterations = NUM_ITERATIONS
# ref model configs
training_args.beta = BETA
training_args.max_grad_norm = MAX_GRAD_NORM
# evals
if TEST_HYPOTHESIS:
    # training_args.eval_strategy = "no"
    # training_args.eval_on_start = False
    training_args.eval_strategy = "steps"
    training_args.eval_on_start = EVAL_ON_START
    training_args.eval_steps = EVAL_STEPS
else:
    training_args.eval_strategy = "steps"
    training_args.eval_on_start = EVAL_ON_START
    training_args.eval_steps = EVAL_STEPS
training_args.save_strategy = "steps"
training_args.save_steps = SAVE_STEPS
training_args.per_device_eval_batch_size = PER_DEVICE_BATCH_SIZE
training_args.eval_accumulation_steps = 1
training_args.data_seed = 42
if UPDATE_REF_MODEL:
    training_args.sync_ref_model = True
    training_args.ref_model_mixup_alpha = 1.0
    training_args.ref_model_sync_steps = SAVE_STEPS

trainer = vf.GRPOEnvTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=rubric,
    env=vf_env,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    debug_generate=DEBUG_GENERATE,
    debug_rewards=DEBUG_REWARDS,
    model_name=model_name,
    run_name=run_name,
    use_dr_grpo=USE_DR_GRPO,
    test_hypothesis_clip_advantage=(TEST_HYPOTHESIS and not BASELINE_RUN),
    apply_overlong_filtering=APPLY_OVERLONG_FILTERING,
)

trainer.train() 