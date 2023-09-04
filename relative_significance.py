from collections import defaultdict


def run_relative_significance(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Sort lines by MSE value
    sorted_lines = sorted(lines, key=lambda line: float(line.split("MSE: ")[1].split(",")[0]))

    # Initialize dictionaries
    param_occurrences = defaultdict(lambda: defaultdict(int))
    param_mse_sum = defaultdict(lambda: defaultdict(float))

    # Extract parameter values and MSE from each line
    for line in sorted_lines:
        parts = line.split(", ")
        mse = float(parts[0].split(": ")[1])
        params = parts[1:]

        for param in params:
            k, v = param.split(": ")
            param_occurrences[k][v] += 1
            param_mse_sum[k][v] += mse

    # Filter out the parameters that were not fully evaluated
    min_occurrences = {}
    for k, v_dict in param_occurrences.items():
        min_occurrences[k] = min(v_dict.values())

    for k, v_dict in param_occurrences.items():
        param_occurrences[k] = {v: count for v, count in v_dict.items() if count == min_occurrences[k]}
        param_mse_sum[k] = {v: mse_sum for v, mse_sum in param_mse_sum[k].items() if v in param_occurrences[k]}

    # Calculate average MSE for each parameter value
    param_avg_mse = {k: {value: param_mse_sum[k][value] / count for value, count in values.items()} for k, values in param_occurrences.items()}

    # Calculate optimality percentage for each parameter value
    param_optimality = {}
    mse_ranges = {}
    total_mse_range = 0

    for k, values in param_avg_mse.items():
        min_mse = min(values.values())
        max_mse = max(values.values())
        mse_range = max_mse - min_mse
        mse_ranges[k] = mse_range
        total_mse_range += mse_range
        param_optimality[k] = {value: 100 * (1 - ((mse - min_mse) / min_mse)) for value, mse in values.items()}

    # Calculate significance percentages for each parameter
    if total_mse_range == 0:
        print("No data available to calculate parameter significance.")
        param_significance = {}
    else:
        param_significance = {k: (mse_ranges[k] / total_mse_range) * 100 for k in mse_ranges}

    # Sort parameters by their significance percentage, in descending order
    sorted_param_significance = {k: v for k, v in sorted(param_significance.items(), key=lambda item: item[1], reverse=True)}

    # Display results
    for k in sorted_param_significance.keys():
        values = param_optimality[k]
        if len(values) > 1:
            print(f"{k} (Significance: {sorted_param_significance[k]:.2f}%)")
            sorted_values = {k: v for k, v in sorted(values.items(), key=lambda item: item[1], reverse=True)}
            for value, optimality in sorted_values.items():
                print(f"  {value}: {optimality:.2f}%")
