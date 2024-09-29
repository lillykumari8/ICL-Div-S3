import argparse
import datetime
import json
import os
import psutil
import random
import time
import warnings
from collections import OrderedDict, defaultdict
from copy import deepcopy

import nltk
import copy
import math
import numpy as np
import openai
import sqlparse
import submarine
import torch
import wandb
from datasets import load_dataset, load_metric
from sentence_transformers import SentenceTransformer
from sklearn.metrics import euclidean_distances, pairwise, pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from torch import nn
from tqdm import tqdm

from MetaICL.metaicl.data import MetaICLData
from MetaICL.metaicl.model import MetaICLModel
from transformers import AutoTokenizer, GPTJForCausalLM
from utils import codex_execution, set_seed, calculate_sentence_transformer_embedding
from get_task import get_dataset
from strategies import *

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description="In-Context Learning")
parser.add_argument("--seed", default=0, type=int)
parser.add_argument(
    "--rnd_seed",
    default=0,
    type=int,
    help="random seed for random sampling when stage_1 is set to stage_1=random",
)

parser.add_argument(
    "--task_name",
    type=str,
    required=False,
    default="sst5",
    choices=["sst5", "mrpc", "dbpedia_14", "mnli", "rte", "hellaswag", "sst2", "trec"],
)
parser.add_argument("--num_train", type=int, default=-1)
parser.add_argument("--num_eval", type=int, default=-1)
parser.add_argument(
    "--model_cache_dir",
    type=str,
    default="model_cache/",
)
parser.add_argument("--data_cache_dir", type=str, default="data/")
parser.add_argument("--output_dir", type=str, default="output/")

parser.add_argument("--model_name", default="EleutherAI/gpt-j-6b", type=str)
parser.add_argument(
    "--embedding_model",
    default="sentence-transformers/paraphrase-mpnet-base-v2",
    type=str,
)

# Stage 1
parser.add_argument(
    "--stage_1",
    required=True,
    type=str,
    choices=["fast-votek", "random", "mfl", "mdiv", "subm-fl-greedy"],
    help="STAGE 1 - method for selecting samples for annotations",
)
# Stage 2
parser.add_argument(
    "--stage_2",
    required=True,
    type=str,
    choices=["similar", "mixmodsub", "random", "sspan"],
    help="STAGE 2 - method for retrieving prompts for each test query.",
)
parser.add_argument(
    "--sspan_div",
    default="greedy",
    type=str,
    choices=["ks_greedy", "greedy"],
    help="Diversifying submdodular span",
)
parser.add_argument(
    "--p_budget",
    default=30,
    type=int,
    help="Final exemplar budget, only applicable if sspan_div is greedy",
)

parser.add_argument("--annotation_size", default=100, type=int)
parser.add_argument("--vote_k", type=int, default=150)

parser.add_argument("--fl_sim_metric", type=str, default="cosine")
parser.add_argument(
    "--fl_sigma",
    type=float,
    default=1.0,
    help="Sigma for RBF kernel for FL similarity matrix.",
)
parser.add_argument(
    "--sspan_nn_factor",
    type=float,
    default=2.0,
    help="NN factor for pre-filtering before sspan (Stage 2)",
)
parser.add_argument(
    "--sspan_k",
    type=int,
    default=40,
    help="Cardinality constraint for sspan (Stage 2)",
)
parser.add_argument(
    "--submflks_gamma",
    type=float,
    default=0.1,
    help="Submodular FL knapsack constraint kcpower / gamma for sspan (Stage 2)",
)
parser.add_argument(
    "--wandb_log", type=str, default="offline", choices=["off", "offline", "online"]
)

# Arguments related to demos/prompts truncation based on position distance
parser.add_argument(
    "--truncate",
    type=int,
    default=0,
    help="if truncate=k=0, then do not truncate prompts, else truncate first k prompts out of selected prompts",
)
parser.add_argument(
    "--truncate_rev",
    type=int,
    default=0,
    help="if truncate_rev, then do not truncate prompts, else \
                        truncate first (num_shots - truncate_rev) prompts out of selected prompts",
)

# mixmodsub stage_2
parser.add_argument(
    "--fl_weight",
    type=float,
    default=0.5,
    help="Weight of FL function in mixture submodular function",
)
args = parser.parse_args()


if __name__ == "__main__":
    time_stamp = str(datetime.datetime.now().isoformat())
    args.exp_name = (
        time_stamp + str(np.random.randint(0, 1000)) + args.stage_1 + "_" + args.stage_2
    )
    if args.wandb_log != "off":
        if args.annotation_size == -1:
            wandb.init(
                project="{}-EVAL-set-non-AL".format(args.task_name), config=args
            )  # non-AL
        else:
            wandb.init(project="{}-EVAL-set".format(args.task_name), config=args)
        wandb.config.update(args)
    else:
        wandb = None

    if args.truncate > 0:
        assert args.truncate_rev == 0, "Cannot have both truncate and truncate_rev > 0"
    if args.truncate_rev > 0:
        assert args.truncate == 0, "Cannot have both truncate and truncate_rev > 0"

    inference_model = MetaICLModel(args=args)
    inference_model.load()
    inference_model.cuda()
    inference_model.eval()

    maximum_input_len = 1000
    single_input_len = 250
    kcbudget = maximum_input_len

    data_module = MetaICLData(
        method="direct", max_length=1024, max_length_per_example=256
    )
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    args.output_dir = (
        args.output_dir
        + args.task_name
        + "/"
        + args.stage_1
        + "_"
        + args.stage_2
        + "/seed_"
        + str(args.seed)
    )
    output_dir = os.path.join(args.output_dir, args.exp_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    prompt_identifier = "prompts"
    prompt_cache_dir = os.path.join(output_dir, prompt_identifier)
    if not os.path.exists(prompt_cache_dir):
        os.makedirs(prompt_cache_dir, exist_ok=True)
    set_seed(args.seed)

    (
        total_train_examples,
        total_eval_examples,
        all_train_text_to_encode,
        all_eval_text_to_encode,
        format_example,
        label_map,
    ) = get_dataset(args)

    print(
        f"num_train: {len(total_train_examples)}, num_eval: {len(total_eval_examples)}"
    )
    total_train_embeds = calculate_sentence_transformer_embedding(
        text_to_encode=all_train_text_to_encode, embedding_model=args.embedding_model
    )
    total_eval_embeds = calculate_sentence_transformer_embedding(
        text_to_encode=all_eval_text_to_encode, embedding_model=args.embedding_model
    )

    if args.annotation_size != -1:
        print(
            "First doing ACTIVE LEARNING to select annotated pool of prompts/demos!!!!!!!"
        )
        if args.stage_1 == "random":
            set_seed(args.rnd_seed)
            stage_1_soln = random.sample(
                range(len(total_train_examples)), args.annotation_size
            )
        elif args.stage_1 == "fast-votek":
            stage_1_soln = fast_votek(
                total_train_embeds,
                args.annotation_size,
                args.vote_k,
                vote_file=os.path.join(output_dir, "nearest_neighbors.json"),
            )
        elif args.stage_1 == "mfl":
            stage_1_soln, fl_set_eval = maximize_fl(
                total_train_embeds, args.annotation_size
            )
        elif args.stage_1 == "mdiv":
            stage_1_soln = maximize_diversity(total_train_embeds, args.annotation_size)
        elif args.stage_1 == "subm-fl-greedy":
            stage_1_soln, fl_set_eval = subm_diversity(
                total_train_embeds,
                args.annotation_size,
                args.fl_sim_metric,
                args.fl_sigma,
            )
        else:
            raise NotImplementedError

        total_train_examples = [total_train_examples[i] for i in stage_1_soln]
        total_train_embeds = total_train_embeds[stage_1_soln, :]
    else:
        args.annotation_size = len(total_train_examples)
        print("Using all training data for annotation!!!!!!!")

    print(
        len(total_train_examples),
        len(total_train_embeds),
        len(all_train_text_to_encode),
    )
    assert args.annotation_size == len(
        total_train_examples
    ), f"annotation_budget={args.annotation_size}, len(total_train_examples)={len(total_train_examples)}"
    assert args.annotation_size == len(
        total_train_embeds
    ), f"annotation_budget={args.annotation_size}, len(total_train_embeds)={len(total_train_embeds)}"

    total_train_examples_costs = []
    for idx in range(len(total_train_examples)):
        cur_example_input_text, cur_example_output_text = format_example(
            example=total_train_examples[idx], label_map=label_map
        )
        cur_len = sum(
            get_instance_length(
                cur_example_input_text, cur_example_output_text, tokenizer=tokenizer
            )
        )
        total_train_examples_costs.append(cur_len)
    total_train_examples_costs = np.array(total_train_examples_costs)
    valid_train_idxs = np.where(total_train_examples_costs <= single_input_len)[0]
    print(
        f"Length of valid training annotated samples: {len(valid_train_idxs)} given annotation bugdet of {args.annotation_size}"
    )

    total_train_examples = [total_train_examples[i] for i in valid_train_idxs]
    total_train_embeds = total_train_embeds[valid_train_idxs, :]
    total_train_examples_costs = total_train_examples_costs[valid_train_idxs]

    # compute similarity matrix
    sim_mat = compute_sims(total_train_embeds, args.fl_sim_metric, args.fl_sigma)
    print(f"Is sim_mat contiguous: {sim_mat.data.contiguous}")

    # modify sim_mat to include test examples
    sim_mat_mod = np.zeros((len(total_train_embeds) + 1, len(total_train_embeds) + 1))
    sim_mat_mod[: len(total_train_embeds), : len(total_train_embeds)] = sim_mat
    sim_mat_mod[-1, -1] = 1.0

    nn_budget = int(args.sspan_nn_factor * args.sspan_k)
    sspan_costs_all = []
    sspan_idxs = []
    final_stage_idxs = []
    prompt_lens = []

    for i in range(len(total_eval_embeds)):
        if args.stage_2 == "similar":
            selected_idxs = []
            one_test_instance_input_text, one_test_instance_output_text = (
                format_example(example=total_eval_examples[i], label_map=label_map)
            )
            cur_prompt_string_len = get_instance_length(
                one_test_instance_input_text, one_test_instance_output_text, tokenizer
            )[0]
            sim_scores = cosine_similarity(
                total_eval_embeds[i].reshape(1, -1), total_train_embeds
            ).reshape(-1)
            sorted_indices = np.argsort(sim_scores)
            num_indices = len(sorted_indices)
            for idx in range(num_indices - 1, -1, -1):
                if sim_scores[sorted_indices[idx]] == 1:
                    continue
                cur_len = total_train_examples_costs[sorted_indices[idx]]
                cur_prompt_string_len += cur_len
                if cur_prompt_string_len > maximum_input_len:
                    break
                selected_idxs.append(sorted_indices[idx])
            final_stage_idxs.append(selected_idxs)
            print(f"Test idx: {i} - prompts count {len(selected_idxs)}")
            prompt_lens.append(len(selected_idxs))

        elif args.stage_2 == "random":
            set_seed(args.rnd_seed + i)
            shuffled_indices = np.random.permutation(len(total_train_examples))
            one_test_instance_input_text, one_test_instance_output_text = (
                format_example(example=total_eval_examples[i], label_map=label_map)
            )
            cur_prompt_string_len = get_instance_length(
                one_test_instance_input_text, one_test_instance_output_text, tokenizer
            )[0]
            selected_idxs = []
            for idx in shuffled_indices:
                cur_len = total_train_examples_costs[idx]
                cur_prompt_string_len += cur_len
                if cur_prompt_string_len > maximum_input_len:
                    break
                selected_idxs.append(idx)

            final_stage_idxs.append(selected_idxs)
            print(f"Test idx: {i} - prompts count {len(selected_idxs)}")
            prompt_lens.append(len(selected_idxs))

        elif args.stage_2 == "sspan":
            test_train_sim_vector = compute_sims_matrix_vector(
                total_train_embeds,
                total_eval_embeds[i],
                args.fl_sim_metric,
                args.fl_sigma,
            ).reshape(-1)
            sim_mat_mod[-1, : len(total_train_embeds)] = test_train_sim_vector
            sim_mat_mod[: len(total_train_embeds), -1] = test_train_sim_vector

            fl = submarine.FacilityLocationFunction(sim_mat_mod)
            grnd_set_size = sim_mat_mod.shape[0]
            q_set = submarine.DefaultSet(set([len(sim_mat)]), grnd_set_size)
            if args.sspan_nn_factor <= 1.0:
                inc_inds = modular_min(
                    fl, q_set, grnd_set_size, reduced_set=[], mode="faq"
                )
            else:
                reduced_set = list(np.argsort(test_train_sim_vector)[-nn_budget:])
                inc_inds = modular_min(
                    fl, q_set, grnd_set_size, reduced_set=reduced_set, mode="faq"
                )
            inc_inds = inc_inds[: args.sspan_k]
            sspan_idxs.append(inc_inds)
            sspan_costs_all.append(total_train_examples_costs[inc_inds])

        elif args.stage_2 == "mixmodsub":
            test_train_sim_vector = compute_sims_matrix_vector(
                total_train_embeds,
                total_eval_embeds[i],
                args.fl_sim_metric,
                args.fl_sigma,
            ).reshape(-1)
            fl_func = submarine.FacilityLocationFunction(sim_mat)
            mod_func = submarine.ModularFunction(
                submarine.VectorFloat(test_train_sim_vector)
            )
            mix_func = submarine.MixtureSubmodularFunction(
                submarine.VectorDouble([args.fl_weight, 1 - args.fl_weight]),
                [fl_func, mod_func],
            )
            summary_size = args.p_budget
            ground_set_size = sim_mat.shape[0]
            solution_set = submarine.DefaultSet(ground_set_size)
            solution_order = np.zeros(summary_size).astype(np.int32)
            solution_gains = np.zeros(summary_size).astype(np.float32)
            submarine.AcceleratedGreedyConstrainedMaximization(
                mix_func, summary_size, solution_set, solution_order, solution_gains
            )
            print(f"Test idx: {i} - sol set eval {mix_func.Evaluate(solution_set)}")
            print(f"Test idx: {i} - greedy gains {solution_gains}")
            final_indices = []
            len_cost = 0
            one_test_instance_input_text, one_test_instance_output_text = (
                format_example(example=total_eval_examples[i], label_map=label_map)
            )
            cur_prompt_string_len = get_instance_length(
                one_test_instance_input_text, one_test_instance_output_text, tokenizer
            )[0]
            kcbudget_mod = kcbudget - cur_prompt_string_len

            for kk in solution_order:
                len_cost += total_train_examples_costs[kk]
                if len_cost > kcbudget_mod:
                    break
                final_indices.append(kk)
            print(f"Test idx: {i} - prompts count {len(final_indices)}")
            prompt_lens.append(len(final_indices))
            final_stage_idxs.append(final_indices)

    if args.stage_2 == "sspan":
        # second stage of submodular span: summarizes the most relevant exemplars
        for test_idx in range(len(total_eval_embeds)):
            sspan_idx = sspan_idxs[test_idx]
            sspan_costs = list(sspan_costs_all[test_idx])

            reduced_feats = total_train_embeds[sspan_idx]
            sim_mat = compute_sims(reduced_feats, args.fl_sim_metric, args.fl_sigma)
            one_test_instance_input_text, one_test_instance_output_text = (
                format_example(
                    example=total_eval_examples[test_idx], label_map=label_map
                )
            )
            cur_prompt_string_len = get_instance_length(
                one_test_instance_input_text, one_test_instance_output_text, tokenizer
            )[0]
            kcbudget_mod = kcbudget - cur_prompt_string_len
            if args.sspan_div == "ks_greedy":
                summary_set = submodular_kc_greedy(
                    sim_mat, sspan_costs, kcbudget_mod, args.submflks_gamma, sspan_costs
                )
            else:
                summary_set = submodular_greedy(
                    sim_mat, kcbudget_mod, args.p_budget, sspan_costs
                )
            soln = summary_set.get_elements()
            final_indices = [sspan_idx[i] for i in soln]
            final_stage_idxs.append(final_indices)
            print(f"Test idx: {test_idx} - prompts count {len(final_indices)}")
            prompt_lens.append(len(final_indices))
    print(f"Average prompts count: {np.mean(prompt_lens)}")

    for test_id, one_test_instance in enumerate(total_eval_examples):
        unsorted_selected_idxs = final_stage_idxs[test_id]
        sim_vector = cosine_similarity(
            total_eval_embeds[test_id].reshape(1, -1),
            total_train_embeds[unsorted_selected_idxs],
        ).reshape(-1)
        sorted_idxs = np.argsort(sim_vector)
        selected_idxs = [unsorted_selected_idxs[i].item() for i in sorted_idxs]

        cur_train_data = []
        if args.truncate > 0:
            selected_idxs = selected_idxs[args.truncate :]
        if args.truncate_rev > 0:
            selected_idxs = selected_idxs[-args.truncate_rev :]
        for idx in selected_idxs:
            cur_input_text, cur_output_text = format_example(
                example=total_train_examples[idx], label_map=label_map
            )
            if args.task_name == "hellaswag":
                cur_train_data.append(
                    {
                        "input": cur_input_text,
                        "output": cur_output_text,
                        "options": total_train_examples[idx]["endings"],
                    }
                )
            else:
                cur_train_data.append(
                    {"input": cur_input_text, "output": cur_output_text}
                )
        second_phase_selected_indices = selected_idxs
        with open(
            os.path.join(prompt_cache_dir, f"{one_test_instance['id']}.json"), "w"
        ) as f:
            json.dump(
                [
                    [
                        test_id,
                        second_phase_selected_indices,
                        one_test_instance["label"],
                    ],
                    cur_train_data,
                    one_test_instance,
                ],
                f,
                indent=4,
            )

    candidate_prompt_files = os.listdir(prompt_cache_dir)
    prompt_files = [f for f in candidate_prompt_files if f.endswith(".json")]
    print(len(candidate_prompt_files), len(prompt_files), prompt_files[:5])

    processed_eval_examples = total_eval_examples
    assert len(prompt_files) == len(processed_eval_examples), (
        f"len(prompt_files)={len(prompt_files)},"
        f"len(processed_eval_examples)={len(processed_eval_examples)}"
    )

    golds = []
    preds = []

    if not args.task_name in ["hellaswag", "xsum", "nq"]:
        all_labels = []
        label_to_digit = {}
        for k, v in label_map.items():
            all_labels.append(v)
            label_to_digit[v] = k
    execution_count = 0

    bar = tqdm(range(len(prompt_files)), desc=f"  LLM inference")
    for file in prompt_files:
        bar.update(1)
        if args.task_name == "hellaswag":
            with open(os.path.join(prompt_cache_dir, file)) as f:
                one_test_example = json.load(f)
            cur_train_data = one_test_example[1]
            cur_input = {
                "input": format_example(
                    one_test_example[2], label_map=label_map, args=args
                )[0],
                "options": one_test_example[2]["endings"],
            }
            data_module.k = len(cur_train_data)
            data_module.tensorize(cur_train_data, [cur_input])
            prediction = inference_model.do_predict(data_module)[0]
            assert prediction in one_test_example[2]["endings"]
            with open(f"{output_dir}/{file}", "w") as f:
                json.dump(
                    [
                        prediction,
                        one_test_example[2]["endings"][one_test_example[2]["label"]],
                    ],
                    f,
                )
            preds.append(prediction)
            golds.append(one_test_example[2]["endings"][one_test_example[2]["label"]])

        elif args.task_name in [
            "sst5",
            "mrpc",
            "dbpedia_14",
            "mnli",
            "rte",
            "sst2",
            "trec",
        ]:
            with open(os.path.join(prompt_cache_dir, file)) as f:
                one_test_example = json.load(f)
            cur_train_data = one_test_example[1]
            for idx in range(len(cur_train_data)):
                cur_train_data[idx]["options"] = all_labels
            cur_input = format_example(one_test_example[2], label_map=label_map)[0]
            data_module.k = len(cur_train_data)
            data_module.tensorize(cur_train_data, [cur_input], options=all_labels)
            prediction = inference_model.do_predict(data_module)[0]
            with open(os.path.join(output_dir, file), "w") as f:
                json.dump([prediction, one_test_example[2]["label"]], f)
            preds.append(label_to_digit[prediction])
            golds.append(one_test_example[2]["label"])

    assert len(golds) == len(preds), f"len(golds)={len(golds)}, len(preds)={len(preds)}"
    total = len(golds)
    correct = 0
    for p, g in zip(golds, preds):
        if p == g:
            correct += 1
    with open(os.path.join(output_dir, "result_summary.txt"), "w") as f:
        f.write(f"{len(golds)} examples, accuracy is: {correct / total}\n")
    print(f"The accuracy is {correct / total}\n")
    if wandb:
        wandb.log({"accuracy": correct / total})
        wandb.finish()
