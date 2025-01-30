from llama_cpp import Llama # is somehow need for windows users
import os
import logging
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from metrics import (
    calculate_bert_score,
    calculate_bleu_nltk,
    calculate_rouge1,
    calculate_rouge2,
    calculate_rougeL
)
from prompt_strategy import (
    ZeroShotStrategy, 
    ChainOfThoughtStrategy, 
    TestStrategy, 
    BaselineStrategy, 
    PersonaStrategy
)
from model_loader import ModelLoader
from plot_creators import (
    create_combined_strategy_comparison_plot,
    create_combined_direction_comparison_plot,
    create_combined_model_ranking_plot,
    create_complexity_overview_plots
)

logger = logging.getLogger(__name__)


def run_pipeline_old(file_path, models, strategies, metrics):
    """
    Runs the entire pipeline:
      - Loads test data
      - For each model and strategy, translates Eng->De and De->Eng
      - Logs metrics per row
      - Logs average metrics across rows
      - Logs artifacts containing translations
      - Logs aggregated averages for each Strategy over all models (top-level)
      - Logs per-model averages (second-level).
    """
    logger.info("Loading test data from %s...", file_path)
    df = pd.read_pickle(file_path)
    logger.debug("Loaded dataframe of shape %s", df.shape)

    with mlflow.start_run(run_name="translation_pipeline"):
        mlflow.log_param("data_file_used", file_path)

        strategy_sums_eng2de = {
            type(strategy).__name__: {m: 0.0 for m in metrics} for strategy in strategies
        }
        strategy_counts_eng2de = {
            type(strategy).__name__: 0 for strategy in strategies
        }
        strategy_sums_de2eng = {
            type(strategy).__name__: {m: 0.0 for m in metrics} for strategy in strategies
        }
        strategy_counts_de2eng = {
            type(strategy).__name__: 0 for strategy in strategies
        }

        for model_dict in models:
            model_name = model_dict["name"]
            model = model_dict["model"]
            logger.info("Starting pipeline for model: %s", model_name)

            # second-level run for each model
            with mlflow.start_run(run_name=model_name, nested=True):
                mlflow.log_param("model_name", model_name)

                model_sums_eng2de = {m: 0.0 for m in metrics}
                model_sums_de2eng = {m: 0.0 for m in metrics}
                model_count_eng2de = 0
                model_count_de2eng = 0

                for strategy in strategies:
                    strategy_name = type(strategy).__name__
                    logger.info("Using strategy: %s", strategy_name)

                    # third-level run for each (model, strategy) pair
                    with mlflow.start_run(run_name=strategy_name, nested=True):
                        mlflow.log_param("model", model_name)
                        mlflow.log_param("strategy", strategy_name)

                        metric_sums_eng2de = {m: 0.0 for m in metrics}
                        metric_sums_de2eng = {m: 0.0 for m in metrics}
                        row_count_eng2de = 0
                        row_count_de2eng = 0

                        translations_for_artifact = []

                        for idx, row in df.iterrows():
                            complexity = row["complexity"]
                            text_english = row["text_english"]
                            text_german = row["text_german"]

                            logger.debug("Processing row idx=%d (complexity=%s)", idx, complexity)
                            mlflow.log_param(f"complexity_{idx}", complexity)

                            # translate English->German
                            translation_eng2de = strategy.translate_to_german(model, text_english)
                            for metric_name, metric_fn in metrics.items():
                                score = metric_fn(text_german, translation_eng2de)

                                mlflow.log_metric(f"{metric_name}_eng2de_{complexity}", score)
                                metric_sums_eng2de[metric_name] += score
                            row_count_eng2de += 1

                            # translate German->English
                            translation_de2eng = strategy.translate_to_english(model, text_german)
                            for metric_name, metric_fn in metrics.items():
                                score = metric_fn(text_english, translation_de2eng)
                                mlflow.log_metric(f"{metric_name}_de2eng_{complexity}", score)
                                metric_sums_de2eng[metric_name] += score
                            row_count_de2eng += 1

                            # collect translations for artifact
                            translations_for_artifact.append(
                                f"Row idx={idx}, complexity={complexity}\n"
                                f"Original English : {text_english}\n"
                                f"Translation Eng->De: {translation_eng2de}\n\n"
                                f"Original German  : {text_german}\n"
                                f"Translation De->Eng: {translation_de2eng}\n\n"
                                "-----------------------------------------------------\n"
                            )

                        # log average metrics for this model and strategy
                        for metric_name in metrics:
                            # Eng->De average
                            if row_count_eng2de > 0:
                                avg_e = metric_sums_eng2de[metric_name] / row_count_eng2de
                                mlflow.log_metric(f"{metric_name}_eng2de_avg", avg_e)
                                logger.debug("Avg %s_eng2de for %s-%s = %.4f", metric_name, model_name, strategy_name, avg_e)

                            # De->Eng average
                            if row_count_de2eng > 0:
                                avg_d = metric_sums_de2eng[metric_name] / row_count_de2eng
                                mlflow.log_metric(f"{metric_name}_de2eng_avg", avg_d)
                                logger.debug("Avg %s_de2eng for %s-%s = %.4f", metric_name, model_name, strategy_name, avg_d)

                            # both directions combined
                            if row_count_eng2de > 0 and row_count_de2eng > 0:
                                both_avg = (
                                    metric_sums_eng2de[metric_name] + metric_sums_de2eng[metric_name]
                                ) / (row_count_eng2de + row_count_de2eng)
                                mlflow.log_metric(f"{metric_name}_both_avg", both_avg)
                                logger.debug("Avg %s_both for %s-%s = %.4f", metric_name, model_name, strategy_name, both_avg)

                        for metric_name in metrics:
                            model_sums_eng2de[metric_name] += metric_sums_eng2de[metric_name]
                            model_sums_de2eng[metric_name] += metric_sums_de2eng[metric_name]

                        model_count_eng2de += row_count_eng2de
                        model_count_de2eng += row_count_de2eng

                        if row_count_eng2de > 0:
                            for metric_name in metrics:
                                strategy_sums_eng2de[strategy_name][metric_name] += metric_sums_eng2de[metric_name]
                            strategy_counts_eng2de[strategy_name] += row_count_eng2de

                        if row_count_de2eng > 0:
                            for metric_name in metrics:
                                strategy_sums_de2eng[strategy_name][metric_name] += metric_sums_de2eng[metric_name]
                            strategy_counts_de2eng[strategy_name] += row_count_de2eng

                        os.makedirs("./logs", exist_ok=True)
                        artifact_file = f"./logs/translations_{model_name}_{strategy_name}.txt"
                        with open(artifact_file, "w", encoding="utf-8") as f:
                            f.writelines(translations_for_artifact)
                        mlflow.log_artifact(artifact_file)
                        logger.info(
                            "Wrote translations artifact for %s / %s to %s",
                            model_name, strategy_name, artifact_file
                        )

                # compute secon-level averages
                for metric_name in metrics:
                    # Eng->De
                    if model_count_eng2de > 0:
                        model_avg_e = model_sums_eng2de[metric_name] / model_count_eng2de
                        mlflow.log_metric(f"{metric_name}_eng2de_model_avg", model_avg_e)
                        logger.info(
                            "[MODEL AVG] %s for Eng->De on model '%s': %.4f (over %d translations)",
                            metric_name, model_name, model_avg_e, model_count_eng2de
                        )

                    # De->Eng
                    if model_count_de2eng > 0:
                        model_avg_d = model_sums_de2eng[metric_name] / model_count_de2eng
                        mlflow.log_metric(f"{metric_name}_de2eng_model_avg", model_avg_d)
                        logger.info(
                            "[MODEL AVG] %s for De->Eng on model '%s': %.4f (over %d translations)",
                            metric_name, model_name, model_avg_d, model_count_de2eng
                        )

                    # both directions combined
                    if model_count_eng2de > 0 and model_count_de2eng > 0:
                        combined = (
                            model_sums_eng2de[metric_name] + model_sums_de2eng[metric_name]
                        ) / (model_count_eng2de + model_count_de2eng)
                        mlflow.log_metric(f"{metric_name}_both_model_avg", combined)
                        logger.info(
                            "[MODEL AVG] %s for both directions on model '%s': %.4f (over %d translations total)",
                            metric_name, model_name, combined, model_count_eng2de + model_count_de2eng
                        )

            # close model
            if hasattr(model, "close"):
                logger.debug("Closing model: %s", model_name)
                model.close()

        # compute strategy-level averages for top-level
        for strategy in strategies:
            s_name = type(strategy).__name__

            # # Eng->De
            # if strategy_counts_eng2de[s_name] > 0:
            #     for metric_name in metrics:
            #         total_score = strategy_sums_eng2de[s_name][metric_name]
            #         total_count = strategy_counts_eng2de[s_name]
            #         avg_score = total_score / total_count
            #         mlflow.log_metric(f"avg_{metric_name}_eng2de_over_all_models_{s_name}", avg_score)
            #         logger.info(
            #             "Strategy %s: Overall average %s_eng2de across all models (total %d translations) = %.4f",
            #             s_name, metric_name, total_count, avg_score
            #         )

            # # De->Eng
            # if strategy_counts_de2eng[s_name] > 0:
            #     for metric_name in metrics:
            #         total_score = strategy_sums_de2eng[s_name][metric_name]
            #         total_count = strategy_counts_de2eng[s_name]
            #         avg_score = total_score / total_count
            #         mlflow.log_metric(f"avg_{metric_name}_de2eng_over_all_models_{s_name}", avg_score)
            #         logger.info(
            #             "Strategy %s: Overall average %s_de2eng across all models (total %d translations) = %.4f",
            #             s_name, metric_name, total_count, avg_score
            #         )

            # both directions combined
            if strategy_counts_eng2de[s_name] > 0 and strategy_counts_de2eng[s_name] > 0:
                for metric_name in metrics:
                    total_e = strategy_sums_eng2de[s_name][metric_name]
                    count_e = strategy_counts_eng2de[s_name]
                    total_d = strategy_sums_de2eng[s_name][metric_name]
                    count_d = strategy_counts_de2eng[s_name]
                    combined_avg = (total_e + total_d) / (count_e + count_d)
                    mlflow.log_metric(f"avg_{metric_name}_both_over_all_models_{s_name}", combined_avg)
                    logger.info(
                        "Strategy %s: Overall avg %s_both across all models (%d) = %.4f",
                        s_name, metric_name, count_e + count_d, combined_avg
                    )
    logger.info("Pipeline run finished.")



def run_pipeline(file_path, models, strategies, metrics):
    """
    Runs the entire pipeline:
      - Loads test data
      - For each model and strategy, translates Eng->De and De->Eng
      - Logs metrics
      - Logs average metrics across rows
      - Logs artifacts (translations)
      - Aggregates final averages, calls plotting functions for overview.
    """

    logger.info("Loading test data from %s...", file_path)
    df = pd.read_pickle(file_path)
    logger.debug("Loaded dataframe of shape %s", df.shape)

    strategy_sums_eng2de = {
        type(strategy).__name__: {m: 0.0 for m in metrics} for strategy in strategies
    }
    strategy_counts_eng2de = {
        type(strategy).__name__: 0 for strategy in strategies
    }
    strategy_sums_de2eng = {
        type(strategy).__name__: {m: 0.0 for m in metrics} for strategy in strategies
    }
    strategy_counts_de2eng = {
        type(strategy).__name__: 0 for strategy in strategies
    }

    # for model ranking plots: 
    model_aggregates_both = {}
    
    # complexity_scores[strategy_name][metric_name]["eng2de"][complexity] = []
    # complexity_scores[strategy_name][metric_name]["de2eng"][complexity] = []
    from collections import defaultdict
    complexity_scores = {}
    for strategy in strategies:
        s_name = type(strategy).__name__
        complexity_scores[s_name] = {}
        for m_name in metrics:
            complexity_scores[s_name][m_name] = {
                "eng2de": defaultdict(list),
                "de2eng": defaultdict(list)
            }

    with mlflow.start_run(run_name="translation_pipeline"):
        mlflow.log_param("data_file_used", file_path)

        for model_dict in models:
            model_name = model_dict["name"]
            model = model_dict["model"]
            logger.info("Starting pipeline for model: %s", model_name)

            with mlflow.start_run(run_name=model_name, nested=True):
                mlflow.log_param("model_name", model_name)

                model_sums_eng2de = {m: 0.0 for m in metrics}
                model_sums_de2eng = {m: 0.0 for m in metrics}
                model_count_eng2de = 0
                model_count_de2eng = 0

                for strategy in strategies:
                    strategy_name = type(strategy).__name__
                    logger.info("Using strategy: %s", strategy_name)

                    with mlflow.start_run(run_name=strategy_name, nested=True):
                        mlflow.log_param("model", model_name)
                        mlflow.log_param("strategy", strategy_name)

                        metric_sums_eng2de = {m: 0.0 for m in metrics}
                        metric_sums_de2eng = {m: 0.0 for m in metrics}
                        row_count_eng2de = 0
                        row_count_de2eng = 0

                        translations_for_artifact = []

                        for idx, row in df.iterrows():
                            complexity = row["complexity"]
                            text_english = row["text_english"]
                            text_german = row["text_german"]

                            mlflow.log_param(f"complexity_{idx}", complexity)

                            # Eng->De 
                            translation_eng2de = strategy.translate_to_german(model, text_english)
                            for metric_name, metric_fn in metrics.items():
                                score = metric_fn(text_german, translation_eng2de)
                                mlflow.log_metric(f"{metric_name}_eng2de_{complexity}", score)
                                metric_sums_eng2de[metric_name] += score
                                # store in complexity_scores
                                complexity_scores[strategy_name][metric_name]["eng2de"][complexity].append(score)

                            row_count_eng2de += 1

                            # De->Eng 
                            translation_de2eng = strategy.translate_to_english(model, text_german)
                            for metric_name, metric_fn in metrics.items():
                                score = metric_fn(text_english, translation_de2eng)
                                mlflow.log_metric(f"{metric_name}_de2eng_{complexity}", score)
                                metric_sums_de2eng[metric_name] += score
                                # store in complexity_scores
                                complexity_scores[strategy_name][metric_name]["de2eng"][complexity].append(score)

                            row_count_de2eng += 1

                            # store translations for artifact
                            translations_for_artifact.append(
                                f"Row idx={idx}, complexity={complexity}\n"
                                f"Original English : {text_english}\n"
                                f"Translation Eng->De: {translation_eng2de}\n\n"
                                f"Original German  : {text_german}\n"
                                f"Translation De->Eng: {translation_de2eng}\n\n"
                                "-----------------------------------------------------\n"
                            )

                        # log average metrics at (model+strategy) level
                        for metric_name in metrics:
                            if row_count_eng2de > 0:
                                avg_e = metric_sums_eng2de[metric_name] / row_count_eng2de
                                mlflow.log_metric(f"{metric_name}_eng2de_avg", avg_e)

                            if row_count_de2eng > 0:
                                avg_d = metric_sums_de2eng[metric_name] / row_count_de2eng
                                mlflow.log_metric(f"{metric_name}_de2eng_avg", avg_d)

                            if row_count_eng2de > 0 and row_count_de2eng > 0:
                                both_avg = (
                                    metric_sums_eng2de[metric_name] + metric_sums_de2eng[metric_name]
                                ) / (row_count_eng2de + row_count_de2eng)
                                mlflow.log_metric(f"{metric_name}_both_avg", both_avg)

                        # accumulate for model-level
                        for metric_name in metrics:
                            model_sums_eng2de[metric_name] += metric_sums_eng2de[metric_name]
                            model_sums_de2eng[metric_name] += metric_sums_de2eng[metric_name]

                        model_count_eng2de += row_count_eng2de
                        model_count_de2eng += row_count_de2eng

                        # accumulate for top-level strategy
                        if row_count_eng2de > 0:
                            for metric_name in metrics:
                                strategy_sums_eng2de[strategy_name][metric_name] += metric_sums_eng2de[metric_name]
                            strategy_counts_eng2de[strategy_name] += row_count_eng2de

                        if row_count_de2eng > 0:
                            for metric_name in metrics:
                                strategy_sums_de2eng[strategy_name][metric_name] += metric_sums_de2eng[metric_name]
                            strategy_counts_de2eng[strategy_name] += row_count_de2eng

                        # save translations artifact
                        os.makedirs("./logs", exist_ok=True)
                        artifact_file = f"./logs/translations_{model_name}_{strategy_name}.txt"
                        with open(artifact_file, "w", encoding="utf-8") as f:
                            f.writelines(translations_for_artifact)
                        mlflow.log_artifact(artifact_file)

                # log model-level averages
                for metric_name in metrics:
                    # Eng->De
                    if model_count_eng2de > 0:
                        model_avg_e = model_sums_eng2de[metric_name] / model_count_eng2de
                        mlflow.log_metric(f"{metric_name}_eng2de_model_avg", model_avg_e)

                    # De->Eng
                    if model_count_de2eng > 0:
                        model_avg_d = model_sums_de2eng[metric_name] / model_count_de2eng
                        mlflow.log_metric(f"{metric_name}_de2eng_model_avg", model_avg_d)

                    # Both
                    if model_count_eng2de > 0 and model_count_de2eng > 0:
                        combined = (
                            model_sums_eng2de[metric_name] + model_sums_de2eng[metric_name]
                        ) / (model_count_eng2de + model_count_de2eng)
                        mlflow.log_metric(f"{metric_name}_both_model_avg", combined)

                        # store for final model ranking plot
                        if metric_name not in model_aggregates_both:
                            model_aggregates_both[metric_name] = {}
                        model_aggregates_both[metric_name][model_name] = combined

            if hasattr(model, "close"):
                model.close()

        # top-level strategy averages
        final_strategy_both = {}  # for strategy comparison plots
        final_strategy_dir = {
            type(strategy).__name__: {m: {"eng2de": 0.0, "de2eng": 0.0} for m in metrics}
            for strategy in strategies
        }

        for strategy in strategies:
            s_name = type(strategy).__name__

            # combined (both directions)
            if (strategy_counts_eng2de[s_name] > 0) and (strategy_counts_de2eng[s_name] > 0):
                final_strategy_both[s_name] = {}
                for metric_name in metrics:
                    total_e = strategy_sums_eng2de[s_name][metric_name]
                    count_e = strategy_counts_eng2de[s_name]
                    total_d = strategy_sums_de2eng[s_name][metric_name]
                    count_d = strategy_counts_de2eng[s_name]

                    combined_avg = (total_e + total_d) / (count_e + count_d)
                    mlflow.log_metric(
                        f"avg_{metric_name}_both_over_all_models_{s_name}",
                        combined_avg
                    )
                    final_strategy_both[s_name][metric_name] = combined_avg

            # for direction-based plot
            e_count = strategy_counts_eng2de[s_name]
            d_count = strategy_counts_de2eng[s_name]
            for metric_name in metrics:
                if e_count > 0:
                    final_strategy_dir[s_name][metric_name]["eng2de"] = \
                        strategy_sums_eng2de[s_name][metric_name] / e_count
                if d_count > 0:
                    final_strategy_dir[s_name][metric_name]["de2eng"] = \
                        strategy_sums_de2eng[s_name][metric_name] / d_count

        # log plots
        create_combined_strategy_comparison_plot(
            final_strategy_both,
            metrics.keys(),
            output_path="plots/01_strategy_comparison.png"
        )

        create_combined_direction_comparison_plot(
            final_strategy_dir,
            metrics.keys(),
            output_path="plots/02_direction_comparison.png"
        )

        # model_aggregates_both = {metric_name: {model_name: combined_score}}
        create_combined_model_ranking_plot(
            model_aggregates_both,
            metrics.keys(),
            output_path="plots/03_model_ranking.png"
        )
        
        # for each strategy, how the scores vary by complexity.
        create_complexity_overview_plots(
            complexity_scores, 
            metrics.keys(), 
            output_dir="plots"
        )


    logger.info("Pipeline run finished.")


if __name__ == "__main__":
    load_dotenv()
    
    logging.basicConfig(level=logging.INFO)  # Shows INFO and WARNING, not DEBUG
    # Or:
    # logging.basicConfig(level=logging.DEBUG)  # Shows DEBUG, INFO, WARNING
    
    file_path = "./data/machine_translation"

    model_path_gemma = os.getenv("MODEL_PATH_GEMMA")
    model_path_llama3_1 = os.getenv("MODEL_PATH_LLAMA3_1")
    model_path_llama3_2 = os.getenv("MODEL_PATH_LLAMA3_2")
    model_path_aya_23 = os.getenv("MODEL_PATH_AYA_23")

    # Load models
    models = [
        ModelLoader.load_llama_cpp_model(model_path_gemma, "gemma-2b-it-q4"),
        ModelLoader.load_llama_cpp_model(model_path_llama3_1, "llama-3.1-8b-q5-k-m"),
        ModelLoader.load_llama_cpp_model(model_path_llama3_2, "llama-3.2-3b-q8-0"),
        # ModelLoader.load_llama_cpp_model(model_path_aya_23, "aya-23-35b-iq2-xxs"),
    ]

    # Strategies
    strategies = [
        BaselineStrategy(),
        PersonaStrategy()
    ]

    # Metrics
    metrics = {
        "BLEU": calculate_bleu_nltk,
        "ROUGE1": calculate_rouge1,
        "ROUGE2": calculate_rouge2,
        "ROUGEL": calculate_rougeL,
        "BERTScore": calculate_bert_score
    }

    mlflow.set_experiment("Dashboard_Experiment") # TODO: change experiment name
    run_pipeline(file_path, models, strategies, metrics)
