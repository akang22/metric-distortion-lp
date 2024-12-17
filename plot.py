import matplotlib.pyplot as plt
from collections import Counter
import math

def approximate_histogram(counter, binsize=1e-2):
    """
    Plots an approximate histogram from a Counter object where keys have small variations.
    
    Args:
        counter (Counter): A Counter object with keys as float values.
        binsize (float): The rounding bin size to group close values (default is 0.01).
    """
    # Group keys based on rounding to the nearest bin
    grouped_counter = Counter()
    for key, count in counter.items():
        rounded_key = round(key / binsize) * binsize  # Round to nearest bin size
        grouped_counter[rounded_key] += count

    # Sort the keys for the histogram
    sorted_keys = sorted(grouped_counter.keys())
    values = [grouped_counter[key] for key in sorted_keys]
    
    # Plot the histogram
    plt.bar(sorted_keys, values, width=binsize * 0.9, align='center', edgecolor="black")
    plt.xlabel("Rounded Values")
    plt.ylabel("Frequency")
    plt.title("Approximate Histogram")
    plt.show()

