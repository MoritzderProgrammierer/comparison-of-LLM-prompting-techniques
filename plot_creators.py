import mlflow
import matplotlib.pyplot as plt
import numpy as np
import os

def create_combined_strategy_comparison_plot(final_strategy_both, metrics, output_path="plots/strategy_comparison.png"):
    """
    Creates ONE figure (PNG) with multiple subplots (one per metric) comparing
    how each strategy performed (both directions combined). Saves + logs to MLflow.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    strategies = list(final_strategy_both.keys())
    metrics = list(metrics) 

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4), squeeze=False)
    axes = axes[0]  # because subplots returns a 2D array

    for ax, metric_name in zip(axes, metrics):
        metric_values = [final_strategy_both[s].get(metric_name, 0.0) for s in strategies]

        ax.bar(strategies, metric_values)
        ax.set_title(f"{metric_name} (Both Directions)")
        ax.set_ylabel("Score")
        ax.set_xticklabels(strategies, rotation=45, ha="right")

    fig.tight_layout()
    plt.savefig(output_path)
    mlflow.log_artifact(output_path)  # Logs to currently-active MLflow run
    plt.close(fig)


def create_combined_direction_comparison_plot(final_strategy_dir, metrics, output_path="plots/direction_comparison.png"):
    """
    Creates ONE figure with multiple subplots (one per metric), each subplot
    shows side-by-side bars for Eng->De vs De->Eng for each strategy.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    strategies = list(final_strategy_dir.keys())
    metrics = list(metrics)
    x = np.arange(len(strategies))
    width = 0.35

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4), squeeze=False)
    axes = axes[0]

    for ax, metric_name in zip(axes, metrics):
        eng2de_scores = [final_strategy_dir[s][metric_name]["eng2de"] for s in strategies]
        de2eng_scores = [final_strategy_dir[s][metric_name]["de2eng"] for s in strategies]

        ax.bar(x - width/2, eng2de_scores, width, label="Eng->De")
        ax.bar(x + width/2, de2eng_scores, width, label="De->Eng")

        ax.set_title(metric_name)
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=45, ha="right")
        ax.set_ylabel("Score")
        ax.legend()

    fig.tight_layout()
    plt.savefig(output_path)
    mlflow.log_artifact(output_path)
    plt.close(fig)


def create_combined_model_ranking_plot(model_aggregates_both, metrics, output_path="plots/model_ranking.png"):
    """
    Creates ONE figure with multiple subplots (one per metric) that ranks models
    by the 'both directions' average for that metric.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    metrics = list(metrics)

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4), squeeze=False)
    axes = axes[0]

    for ax, metric_name in zip(axes, metrics):
        model_scores = model_aggregates_both.get(metric_name, {})
        sorted_items = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        model_names = [m for (m, _) in sorted_items]
        values = [v for (_, v) in sorted_items]

        ax.bar(model_names, values)
        ax.set_title(f"{metric_name} (Both Directions)")
        ax.set_ylabel("Score")
        ax.set_xticklabels(model_names, rotation=45, ha="right")

    fig.tight_layout()
    plt.savefig(output_path)
    mlflow.log_artifact(output_path)
    plt.close(fig)

def create_complexity_overview_plots(complexity_scores, metrics, output_dir="plots"):
    """
    Creates a figure (or multiple figures) that shows how translation quality
    changes with complexity. For each metric, we produce one PNG with 2 subplots:
    - Left subplot: Eng->De
    - Right subplot: De->Eng
    Each subplot has one line per strategy, with complexities on the X-axis.
    """
    os.makedirs(output_dir, exist_ok=True)

    all_complexities = set()
    for s_name, s_data in complexity_scores.items():
        for m_name, m_data in s_data.items():
            for cplx in m_data["eng2de"].keys():
                all_complexities.add(cplx)
            for cplx in m_data["de2eng"].keys():
                all_complexities.add(cplx)


    complexities_sorted = sorted(all_complexities, key=str)

    # for each metric 1 figure with 2 subplots
    for metric_name in metrics:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        ax_eng2de, ax_de2eng = axes

        strategies = list(complexity_scores.keys())

        # Eng->De subplot
        for s_name in strategies:
            y_vals = []
            for cplx in complexities_sorted:
                scores_list = complexity_scores[s_name][metric_name]["eng2de"].get(cplx, [])
                if scores_list:
                    avg_score = sum(scores_list) / len(scores_list)
                else:
                    avg_score = 0.0
                y_vals.append(avg_score)

            ax_eng2de.plot(complexities_sorted, y_vals, marker='o', label=s_name)
        
        ax_eng2de.set_title(f"{metric_name} (Eng->De)")
        ax_eng2de.set_xlabel("Complexity")
        ax_eng2de.set_ylabel("Score")
        ax_eng2de.legend(loc="lower left", bbox_to_anchor=(1,1))

        # De->Eng subplot
        for s_name in strategies:
            y_vals = []
            for cplx in complexities_sorted:
                scores_list = complexity_scores[s_name][metric_name]["de2eng"].get(cplx, [])
                if scores_list:
                    avg_score = sum(scores_list) / len(scores_list)
                else:
                    avg_score = 0.0
                y_vals.append(avg_score)

            ax_de2eng.plot(complexities_sorted, y_vals, marker='o', label=s_name)

        ax_de2eng.set_title(f"{metric_name} (De->Eng)")
        ax_de2eng.set_xlabel("Complexity")

        fig.suptitle(f"Complexity vs. Scores for {metric_name}", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.95])  # to make room for suptitle

        outpath = os.path.join(output_dir, f"complexity_overview_{metric_name}.png")
        plt.savefig(outpath)
        mlflow.log_artifact(outpath)
        plt.close(fig)
