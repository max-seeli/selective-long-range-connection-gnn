import matplotlib.pyplot as plt


def extract_training_data(filename):
    """
    Extracts training data from a file and returns a dictionary with the structure:
    {epoch: [], train_loss: [], train_acc: [], test_acc: []}

    Parameters:
    ----------
    filename (str): The name of the file containing the training data.

    Returns:
    -------
    dict: A dictionary containing the extracted data.
    """
    import re

    # Define the regex pattern
    pattern = r"Epoch (\d+), LR: \[\d+\.\d+\]: Train loss: (\d+\.\d+), Train acc: (\d+\.\d+), Test accuracy: (\d+\.\d+)"

    # Initialize the dictionary
    extracted_dict = {"epoch": [], "train_loss": [], "train_acc": [], "test_acc": []}

    try:
        # Open the file
        with open(filename, 'r') as file:
            # Read the file line by line
            for line in file:
                # Search for the pattern in each line
                match = re.search(pattern, line)
                if match:
                    # Extract and append the data
                    epoch, train_loss, train_acc, test_acc = match.groups()
                    extracted_dict["epoch"].append(int(epoch))
                    extracted_dict["train_loss"].append(float(train_loss))
                    extracted_dict["train_acc"].append(float(train_acc))
                    extracted_dict["test_acc"].append(float(test_acc))
    except FileNotFoundError:
        print(f"File {filename} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return extracted_dict


import matplotlib.pyplot as plt


def plot_multiple_training_data(extracted_dicts, titles, save_path=None):
    """
    Plots multiple extracted training data sets side by side.

    Parameters:
    ----------
    extracted_dicts (list of dict): A list of dictionaries, each containing the extracted data for a different set.
    titles (list of str): A list of titles for each subplot.
    save_path (str, optional): The path to save the plot to.
    """
    n = len(extracted_dicts)
    fig, axes = plt.subplots(1, n, figsize=(8*n, 5))  # Adjust the figsize as needed

    for i, extracted_dict in enumerate(extracted_dicts):
        if n > 1:
            ax1 = axes[i]
        else:
            ax1 = axes

        ax2 = ax1.twinx()

        # Plot the training and test accuracy
        ax1.plot(extracted_dict["epoch"], extracted_dict["train_acc"], label="Train accuracy", color="blue")
        ax1.plot(extracted_dict["epoch"], extracted_dict["test_acc"], label="Test accuracy", color="orange")

        # Plot the training loss
        ax2.plot(extracted_dict["epoch"], extracted_dict["train_loss"], label="Train loss", color="green")

        # Set the labels and title
        ax1.set_xlabel("Epoch", fontsize=14)
        ax1.set_ylabel("Accuracy", fontsize=14)
        ax2.set_ylabel("Loss", fontsize=14)
        ax1.set_title(titles[i], fontsize=16)

        # Find the epoch with the maximum test accuracy
        max_test_acc = max(extracted_dict["test_acc"])
        max_test_acc_epoch = extracted_dict["epoch"][extracted_dict["test_acc"].index(max_test_acc)]
        ax1.scatter(max_test_acc_epoch, max_test_acc, color="red", zorder=5)
        ax1.annotate(f"Max test acc: {max_test_acc:.4f} @ epoch {max_test_acc_epoch}", xy=(max_test_acc_epoch, max_test_acc), xytext=(0, 0),
                     textcoords="offset points", ha="left", va="top", rotation=-90, fontsize=12)

        # Combining legends
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc=2, fontsize=14)

        # Font of the ticks
        ax1.tick_params(axis="both", labelsize=12)
        ax2.tick_params(axis="both", labelsize=12)

    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.4)

    # Save the plot
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", format="pdf")
        
    plt.show()



def plot_training_data_from_files(filenames, titles, save_path=None):
    """
    Extracts and plots the training data from files.

    Parameters:
    ----------
    filenames (list of str): A list of file names.
    titles (list of str): A list of titles for each subplot.
    save_path (str): The path to save the plot to.
    """
    extracted_dicts = [extract_training_data(filename) for filename in filenames]
    plot_multiple_training_data(extracted_dicts, titles, save_path)


filenameGCN = "results/gcn_khop_d4.txt"
titleGCN = "GCN with K-Hop transformation, depth 4"
filenameGIN = "results/gin_khop_d4.txt"
titleGIN = "GIN with K-Hop transformation, depth 4"

save_path = "img/learning_curves.pdf"
plot_training_data_from_files([filenameGCN, filenameGIN], [titleGCN, titleGIN], save_path)

