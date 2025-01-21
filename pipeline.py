import os
import logging
import mlflow
import pandas as pd
from dotenv import load_dotenv
from metrics import (
    calculate_bert_score,
    calculate_bleu_nltk,
    calculate_rouge1,
    calculate_rouge2,
    calculate_rougeL
)
from prompt_strategy import ZeroShotStrategy, ChainOfThoughtStrategy, TestStrategy
from model_loader import ModelLoader

logger = logging.getLogger(__name__)

def run_pipeline(file_path, models, strategies, metrics):
    """
    Runs the entire pipeline:
      - Loads test data
      - For each model and strategy, translates Eng->De and De->Eng
      - Logs metrics per row
      - Logs average metrics across rows
      - Logs artifacts containing translations
    """

    logger.info("Loading test data from %s...", file_path)
    df = pd.read_pickle(file_path)
    logger.debug("Loaded dataframe of shape %s", df.shape)

    # Start one MLflow run for the entire pipeline
    with mlflow.start_run(run_name="translation_pipeline"):
        mlflow.log_param("data_file_used", file_path)

        for model_dict in models:
            model_name = model_dict["name"]
            model = model_dict["model"]
            logger.info("Starting pipeline for model: %s", model_name)

            mlflow.log_param("model_loaded", model_name)

            for strategy in strategies:
                strategy_name = type(strategy).__name__
                logger.info("Using strategy: %s", strategy_name)

                # Nested MLflow run for each (model, strategy) pair
                with mlflow.start_run(run_name=f"{model_name}_{strategy_name}", nested=True):
                    mlflow.log_param("model", model_name)
                    mlflow.log_param("strategy", strategy_name)

                    # Sums for computing averages
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

                        # 1) Translate English -> German
                        translation_eng2de = strategy.translate_to_german(model, text_english)
                        logger.debug(
                            "English->German translation for row %d: %r",
                            idx, translation_eng2de
                        )

                        # Calculate metrics Eng->De
                        for metric_name, metric_fn in metrics.items():
                            score = metric_fn(text_german, translation_eng2de)
                            mlflow.log_metric(f"{metric_name}_eng2de_{complexity}", score)
                            metric_sums_eng2de[metric_name] += score
                        row_count_eng2de += 1

                        # 2) Translate German -> English
                        translation_de2eng = strategy.translate_to_english(model, text_german)
                        logger.debug(
                            "German->English translation for row %d: %r",
                            idx, translation_de2eng
                        )

                        # Calculate metrics De->Eng
                        for metric_name, metric_fn in metrics.items():
                            score = metric_fn(text_english, translation_de2eng)
                            mlflow.log_metric(f"{metric_name}_de2eng_{complexity}", score)
                            metric_sums_de2eng[metric_name] += score
                        row_count_de2eng += 1

                        # Collect translations for artifact
                        translations_for_artifact.append(
                            f"Row idx={idx}, complexity={complexity}\n"
                            f"Original English : {text_english}\n"
                            f"Translation Eng->De: {translation_eng2de}\n\n"
                            f"Original German  : {text_german}\n"
                            f"Translation De->Eng: {translation_de2eng}\n\n"
                            "-----------------------------------------------------\n"
                        )

                    # After processing all rows, log average metrics
                    if row_count_eng2de > 0:
                        for metric_name in metrics:
                            avg_score = metric_sums_eng2de[metric_name] / row_count_eng2de
                            mlflow.log_metric(f"{metric_name}_eng2de_avg", avg_score)
                            logger.debug(
                                "Average %s_eng2de across %d rows: %f",
                                metric_name, row_count_eng2de, avg_score
                            )

                    if row_count_de2eng > 0:
                        for metric_name in metrics:
                            avg_score = metric_sums_de2eng[metric_name] / row_count_de2eng
                            mlflow.log_metric(f"{metric_name}_de2eng_avg", avg_score)
                            logger.debug(
                                "Average %s_de2eng across %d rows: %f",
                                metric_name, row_count_de2eng, avg_score
                            )

                    # Write translations to an artifact file
                    os.makedirs("./logs", exist_ok=True)
                    artifact_file = f"./logs/translations_{model_name}_{strategy_name}.txt"
                    with open(artifact_file, "w", encoding="utf-8") as f:
                        f.writelines(translations_for_artifact)
                    mlflow.log_artifact(artifact_file)
                    logger.info(
                        "Wrote translations artifact for %s / %s to %s",
                        model_name, strategy_name, artifact_file
                    )

            # Close model if needed, does not properly work right now.
            if hasattr(model, "close"):
                logger.debug("Closing model: %s", model_name)
                model.close()

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
        ModelLoader.load_llama_cpp_model(model_path_gemma, "gemma-2b-q8"),
        # ModelLoader.load_llama_cpp_model(model_path_llama3_1, "llama-3.1-8b-q5-k-m"),
        # ModelLoader.load_llama_cpp_model(model_path_llama3_2, "llama-3.2-3b-q8-0"),
        # ModelLoader.load_llama_cpp_model(model_path_aya_23, "aya-23-35b-iq2-xxs"),
    ]

    # Strategies
    strategies = [
        ZeroShotStrategy(),
        ChainOfThoughtStrategy(),
        # TestStrategy()
    ]

    # Metrics
    metrics = {
        "BLEU": calculate_bleu_nltk,
        "ROUGE1": calculate_rouge1,
        "ROUGE2": calculate_rouge2,
        "ROUGEL": calculate_rougeL,
        "BERTScore": calculate_bert_score
    }

    run_pipeline(file_path, models, strategies, metrics)
