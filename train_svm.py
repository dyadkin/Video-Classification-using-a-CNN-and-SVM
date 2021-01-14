from sklearn.svm import SVC
from sklearn import datasets
from data import DataSet
from joblib import dump, load
import numpy as np
import time


def test(saved_model, data, data_type, seq_length):
    X, y = data.get_all_sequences_in_memory("test", data_type)
    print(X.shape)
    print(y.shape)
    X = np.reshape(X, (X.shape[0] * seq_length, X.shape[2]))
    print(X.shape)
    clf = load(saved_model)
    print(clf)
    # return clf.predict(X)
    y = np.repeat(np.argmax(y, axis=1), seq_length)
    return clf.score(X, y)


def main():
    # Defaults
    # batch_size = 1
    data_type = "features"
    seq_length = 40
    class_limit = 10
    saved_model = "./data/svm_checkpoints/svc-1609134835.1495786.joblib"

    # X, y = datasets.load_iris(return_X_y=True)
    # print(X.shape)
    # print(y.shape)
    # print(y)
    data = DataSet(seq_length=seq_length, class_limit=class_limit)
    # # path = './data/sequences/v_ApplyEyeMakeup_g01_c01-40-features.npy'
    # # feature = np.load(path)
    # # print(feature)
    # generator = data.frame_generator(batch_size, "train", data_type)
    # for item in generator:
    #     print(item[0].shape)
    #     clf = SVC(decision_function_shape='ovo')
    #     print(clf.kernel)
    #     print(item[1])
    #     print(np.tile(np.squeeze(item[1], axis=0), (item[0].shape[1],)))
    #     # clf.fit(np.squeeze(item[0], axis=0), np.tile(np.squeeze(item[1], axis=0), (item[0].shape[1],1)))
    #     break
    if saved_model:
        pred = test(saved_model, data, data_type, seq_length)
        print(pred)
        timestamp = time.time()
        # pred_path = "./data/svm_checkpoints/pred-" + str(timestamp) + ".npy"
        # np.save(pred_path, pred)
    else:
        X, y = data.get_all_sequences_in_memory("train", data_type)
        print("X shape and byte size:", X.shape, X.nbytes)
        X = np.reshape(X, (X.shape[0] * seq_length, X.shape[2]))
        print("X reshaped:", X.shape)

        print("y shape:", y.shape)
        # Change Y from one hot to index and repeat every row seq len times
        y = np.repeat(np.argmax(y, axis=1), seq_length)
        print("y indexed and reapeated:", y.shape)

        timestamp = time.time()
        clf = SVC(decision_function_shape="ovo", verbose=True)
        print(clf.kernel)
        clf.fit(X, y)
        dump(clf, "./data/svm_checkpoints/svc-" + str(timestamp) + ".joblib")


if __name__ == "__main__":
    main()
