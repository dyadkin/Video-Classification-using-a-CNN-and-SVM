"""
Given a training log file, plot something.
"""
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def main(training_log):
    with open(training_log) as fin:
        reader = csv.reader(fin)
        metrics = next(reader)

        data = {m: [] for m in metrics}
        for row in reader:
            for i in range(len(metrics)):
                data[metrics[i]].append(float(row[i]))

        plt.figure(figsize=(8, 12))
        for m in metrics[1:]:
            plt.plot(data[m], label=m)

        ax = plt.gca()
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
        plt.grid(True)
        plt.xlabel("Epochs")
        plt.title("Custom Net Training")
        plt.legend()
        plt.savefig("trainlog_plot.png")


if __name__ == "__main__":
    training_log = "data/logs/SF_MultiRes-training-1607744970.601056.log"
    main(training_log)
