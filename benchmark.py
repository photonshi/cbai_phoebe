# benchmark a language model on the dataset using zero-shot inference
# use smaller model, same model for all tasks
# read dataset from file dataset.txt

import openai
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import defaultdict
import re # Added for parsing answers

def read_dataset(file_path:str):
    """
    read the dataset from the file
    dataset is a list of dictionaries
    """
    with open(file_path, "r") as f:
        return json.load(f)

def generate_question(example:dict):
    """
    generate a question for the model to answer
    """
    type, list_items, answer = example["type"], example["list"], example["answer"]
    question = f"Count the number of words in the following list that match the given type, and put the numerical answer in parentheses. Output only the answer in parentheses."
    question+=(f"\\n Type: {type}\\n List: {list_items}")
    return question, f"({answer})"

def parse_answer_value(answer_str: str):
    """
    Parses an answer string like \"(3)\" into an integer.
    Returns the integer if successful, None otherwise.
    """
    match = re.fullmatch(r"\((\d+)\)", answer_str)
    if match:
        return int(match.group(1))
    return None

def benchmark_model(model:str, dataset:list):
    """
    Benchmark a language model, providing detailed error analysis per category.
    """
    client = openai.OpenAI()
    
    categories = sorted(list(set(example["type"] for example in dataset)))
    
    category_stats = defaultdict(lambda: {
        "correct": 0, "formatting_error": 0, 
        "off_by_one_error": 0, "large_error": 0, 
        "total": 0, "error_magnitudes": []
    })

    overall_summary = {"correct": 0, "formatting_error": 0, "off_by_one_error": 0, "large_error": 0, "total": 0}

    for example in dataset:
        question, true_answer_str = generate_question(example)
        true_category = example["type"]
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": question}],
            temperature=0,
            max_tokens=10, 
        )
        model_answer_str = response.choices[0].message.content.strip()
        
        print(f"Category: {true_category}, Model: {model_answer_str}, True: {true_answer_str}")

        category_stats[true_category]["total"] += 1
        overall_summary["total"] += 1
        
        true_value = parse_answer_value(true_answer_str) # Should always parse as we generate it

        if model_answer_str == true_answer_str:
            category_stats[true_category]["correct"] += 1
            overall_summary["correct"] += 1
        else:
            model_value = parse_answer_value(model_answer_str)
            if model_value is None:
                category_stats[true_category]["formatting_error"] += 1
                overall_summary["formatting_error"] += 1
            else:
                # It's a numerical answer, but not matching the string (e.g. (03) vs (3) or wrong number)
                # Re-check with numerical equality if string equality failed but format is okay
                if model_value == true_value: # Handles cases like (03) vs (3) if string match failed
                    category_stats[true_category]["correct"] += 1
                    overall_summary["correct"] +=1
                else: # Numerically incorrect
                    error_magnitude = abs(model_value - true_value)
                    category_stats[true_category]["error_magnitudes"].append(model_value - true_value)
                    if error_magnitude == 1:
                        category_stats[true_category]["off_by_one_error"] += 1
                        overall_summary["off_by_one_error"] += 1
                    else:
                        category_stats[true_category]["large_error"] += 1
                        overall_summary["large_error"] += 1
    
    print("\\n--- Detailed Category Performance ---")
    for category in categories:
        stats = category_stats[category]
        accuracy = (stats["correct"] / stats["total"]) if stats["total"] > 0 else 0
        print(f"Category: {category}")
        print(f"  Total: {stats['total']}")
        print(f"  Correct: {stats['correct']} (Accuracy: {accuracy:.2f})")
        print(f"  Formatting Errors: {stats['formatting_error']}")
        print(f"  Off-by-One Errors: {stats['off_by_one_error']}")
        print(f"  Large Errors (>1): {stats['large_error']}")
        if stats["error_magnitudes"]:
            avg_error_magnitude = np.mean(np.abs(stats["error_magnitudes"]))
            bias = np.mean(stats["error_magnitudes"]) # Positive: overcount, Negative: undercount
            print(f"  Avg. Abs. Error Magnitude (for numerical errors): {avg_error_magnitude:.2f}")
            print(f"  Error Bias (model - true): {bias:.2f}")

    print("\\n--- Overall Summary ---")
    total_processed = overall_summary["total"]
    if total_processed > 0:
        print(f"Total Examples: {total_processed}")
        print(f"  Overall Correct: {overall_summary['correct']} ({(overall_summary['correct']/total_processed)*100:.2f}%)")
        print(f"  Overall Formatting Errors: {overall_summary['formatting_error']} ({(overall_summary['formatting_error']/total_processed)*100:.2f}%)")
        print(f"  Overall Off-by-One Errors: {overall_summary['off_by_one_error']} ({(overall_summary['off_by_one_error']/total_processed)*100:.2f}%)")
        print(f"  Overall Large Errors: {overall_summary['large_error']} ({(overall_summary['large_error']/total_processed)*100:.2f}%)")

    return {
        "category_stats": dict(category_stats), # Convert defaultdict for easier use later
        "overall_summary": overall_summary,
        "categories_list": categories
    }

def visualize_results(benchmark_results:dict, model_name:str):
    """
    Visualize the detailed benchmark results, focusing on error types per category.
    """
    category_stats = benchmark_results["category_stats"]
    categories = benchmark_results["categories_list"]
    
    if not categories:
        print("No categories to visualize.")
        return

    n_categories = len(categories)
    
    correct_counts = np.array([category_stats[cat].get("correct", 0) for cat in categories])
    formatting_errors = np.array([category_stats[cat].get("formatting_error", 0) for cat in categories])
    off_by_one_errors = np.array([category_stats[cat].get("off_by_one_error", 0) for cat in categories])
    large_errors = np.array([category_stats[cat].get("large_error", 0) for cat in categories])
    
    # Plot 1: Stacked Bar Chart for Error Types per Category
    plt.figure(figsize=(12, 8))
    bar_width = 0.6
    
    plt.bar(categories, correct_counts, bar_width, label='Correct')
    plt.bar(categories, formatting_errors, bar_width, bottom=correct_counts, label='Formatting Error')
    plt.bar(categories, off_by_one_errors, bar_width, bottom=correct_counts + formatting_errors, label='Off-by-One Error')
    plt.bar(categories, large_errors, bar_width, bottom=correct_counts + formatting_errors + off_by_one_errors, label='Large Error (>1)')
    
    plt.xlabel("Category")
    plt.ylabel("Number of Examples")
    plt.title("Zero-Shot Benchmark: Outcome Types by Category")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{model_name}_zero_shot_outcome_types_by_category.png")
    print("Saved outcome types plot to zero_shot_outcome_types_by_category.png")
    plt.show()

    # Plot 2: Per-category accuracy (as a simple bar chart for clarity)
    accuracies = [(category_stats[cat]["correct"] / category_stats[cat]["total"]) if category_stats[cat]["total"] > 0 else 0 for cat in categories]
    plt.figure(figsize=(12, 7))
    plt.bar(categories, accuracies, color='skyblue')
    plt.xlabel("Category")
    plt.ylabel("Accuracy (Correct / Total)")
    plt.title("Zero-Shot Benchmark: Accuracy by Category")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.02, f"{acc:.2f}", ha='center')
    plt.tight_layout()
    plt.savefig(f"{model_name}_zero_shot_category_accuracy_detailed.png")
    print("Saved category accuracy plot to zero_shot_category_accuracy_detailed.png")
    plt.show()
    
    # Plot 3: Distribution of error magnitudes (e.g., box plot or violin plot per category)
    # Collect all numerical error magnitudes by category
    error_magnitude_data = []
    plot_categories_for_errors = []
    for cat in categories:
        magnitudes = category_stats[cat].get("error_magnitudes", [])
        if magnitudes: # Only include categories with numerical errors
            error_magnitude_data.extend([(cat, err) for err in magnitudes])
            if cat not in plot_categories_for_errors:
                 plot_categories_for_errors.append(cat)


    if error_magnitude_data:
        # Create a DataFrame for easier plotting with seaborn
        error_df_list = [{'category': item[0], 'error_magnitude': item[1]} for item in error_magnitude_data]
        error_df = sns.utils.pd.DataFrame(error_df_list)


        plt.figure(figsize=(12, 7))
        sns.boxplot(x='category', y='error_magnitude', data=error_df, order=plot_categories_for_errors)
        plt.axhline(0, color='red', linestyle='--', lw=0.8) # Line at zero error
        plt.xlabel("Category")
        plt.ylabel("Error Magnitude (Model Count - True Count)")
        plt.title("Distribution of Numerical Error Magnitudes by Category")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(f"{model_name}_zero_shot_error_magnitudes.png")
        print("Saved error magnitudes plot to zero_shot_error_magnitudes.png")
        plt.show()
    else:
        print("No numerical errors with magnitudes recorded to plot.")


if __name__ == "__main__":
    dataset = read_dataset("dataset.json")
    results = benchmark_model("gpt-4o", dataset)
    if results and results["categories_list"]: # Ensure there are categories
        visualize_results(results, "gpt-4o")
    elif not results["categories_list"]:
        print("No categories found in the dataset. Skipping visualization.")
    else:
        print("Benchmarking failed or produced no results.")
