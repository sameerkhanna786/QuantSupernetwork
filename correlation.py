from scipy.stats import spearmanr, kendalltau
import matplotlib.pyplot as plt

def kendall_tau_correlation(list1, list2):
    """
    Calculate the Kendall Tau correlation between two lists of rankings.

    Parameters:
    list1 (list): The first list of rankings.
    list2 (list): The second list of rankings.

    Returns:
    float: The Kendall Tau correlation coefficient.
    """
    return kendalltau(list1, list2).correlation

def spearman_correlation(list1, list2):
    """
    Calculate the Spearman correlation between two lists of rankings.

    Parameters:
    list1 (list): The first list of rankings.
    list2 (list): The second list of rankings.

    Returns:
    float: The Spearman correlation coefficient.
    """
    return spearmanr(list1, list2).correlation

def string_to_list(string):
    """
    Convert a comma-separated string of numbers into a list of numbers.

    Parameters:
    string (str): The comma-separated string.

    Returns:
    list: The list of numbers.
    """
    return [int(x) for x in string.split(',')]

def plot_correlations(spears, kendals, i):
    """
    Plot the Kendall Tau and Spearman correlation rankings across epochs.

    Parameters:
    spears (list): List of Spearman correlation coefficients across epochs.
    kendals (list): List of Kendall Tau correlation coefficients across epochs.

    Returns:
    None
    """
    epochs = range(1, len(spears) + 1)
    plt.plot(epochs, spears, label='Spearman Correlation')
    plt.plot(epochs, kendals, label='Kendall Tau Correlation')
    plt.xlabel('Epochs')
    plt.ylabel('Correlation Coefficient')
    plt.title('Correlation of Rankings Across Epochs (QAT)')
    plt.legend()
    plt.savefig(f"correl_plot_qat_{i}.pdf")
    plt.clf()

def running_average(data, window_size):
    """
    Smooth out a list of values using a running average.

    Parameters:
    data (list): The list of values.
    window_size (int): The size of the window for the running average.

    Returns:
    list: The smoothed list of values.
    """
    smoothed_data = []
    for i in range(len(data)):
        window = data[max(0, i - window_size + 1):i + 1]
        smoothed_data.append(sum(window) / len(window))
    return smoothed_data

def max_pooling(data, window_size):
    """
    Smooth out a list of values using max pooling.

    Parameters:
    data (list): The list of values.
    window_size (int): The size of the window for max pooling.

    Returns:
    list: The smoothed list of values.
    """
    smoothed_data = []
    for i in range(len(data)):
        window = data[max(0, i - window_size + 1):i + 1]
        smoothed_data.append(max(window))
    return smoothed_data

first = "experiment/ranks_file.txt"
second = "experiment/ranks_file_quant.txt"

with open(first) as f:
    first_lines = [line.rstrip('\n') for line in f]

with open(second) as f:
    sec_lines = [line.rstrip('\n') for line in f]

spears = []
kendals = []
for j in range(len(first_lines)):
	cur_spear = 0.0
	cur_ken = 0.0
	for i in range(4):
		cur_list = string_to_list(first_lines[j].split("|")[i])
		sec_list = string_to_list(sec_lines[j].split("|")[i])
		spear = spearman_correlation(cur_list, sec_list)
		ken = kendall_tau_correlation(cur_list, sec_list)

		if spear > cur_spear:
			cur_spear = spear

		if ken > cur_ken:
			cur_ken = ken

	spears.append(cur_spear)
	kendals.append(cur_ken)

spears = spears[2:]
kendals = kendals[2:]
spears = max_pooling(spears, 3)
kendals = max_pooling(kendals, 3)
spears = running_average(spears, 5)
kendals = running_average(kendals, 5)
print(spears[:10])
print(kendals[:10])
plot_correlations(spears, kendals, 0)
