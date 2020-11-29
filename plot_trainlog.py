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
        #losses = []
        top_5_accuracies = []
        cnn_benchmark = []  # this is ridiculous
        for epoch,acc,loss,top_k_categorical_accuracy in reader: #val_acc,val_loss ,val_top_k_categorical_accuracy in reader:
            accuracies.append(float(acc))
            #losses.append(float(loss))
            top_5_accuracies.append(float(top_k_categorical_accuracy))
            cnn_benchmark.append(0.65)  # ridiculous

        plt.plot(accuracies, label='accuracy')
        #plt.plot(losses)
        plt.plot(top_5_accuracies, label='Top 5 accuracy')
        plt.plot(cnn_benchmark, label='0.65 benchmark')
        plt.xlabel('Epochs')
        plt.title('Custom Net Training')
        plt.legend()
        plt.savefig('trainlog_plot.png')

if __name__ == '__main__':
    training_log = 'data/logs/SF_MultiRes-training-1606633512.1547835.log'
    main(training_log)
