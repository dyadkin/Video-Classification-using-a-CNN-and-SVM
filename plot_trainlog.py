"""
Given a training log file, plot something.
"""
import csv
import matplotlib.pyplot as plt

def main(training_log):
    with open(training_log) as fin:
        reader = csv.reader(fin)
        next(reader, None)  # skip the header
        accuracies = []
        top_5_accuracies = []
        cnn_benchmark = []  # this is ridiculous
        for epoch,acc,loss,top_k_categorical_accuracy,val_acc,val_loss,val_top_k_categorical_accuracy in reader:
            accuracies.append(float(val_acc))
            top_5_accuracies.append(float(val_top_k_categorical_accuracy))
            cnn_benchmark.append(0.65)  # ridiculous

        plt.plot(accuracies)
        plt.plot(top_5_accuracies)
        plt.plot(cnn_benchmark)
        plt.savefig('trainlog_plot.png')

if __name__ == '__main__':
    training_log = 'data/logs/InceptionV3-training-1606265533.4140508.log'
    main(training_log)
