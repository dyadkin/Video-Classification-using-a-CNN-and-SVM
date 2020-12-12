# import os
import tensorflow as tf

# from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# from multiprocessing import Pool


def compute_dataset_mean(data_iterator, batch_size):
    sum_mean = 0
    batches = 1
    for x, y in data_iterator:
        if batches == 55888:
            break
        else:
            mean = tf.math.reduce_mean(x)
            print(mean)
            sum_mean += mean
            print(batches)
            batches += 1
    return sum_mean / batches * batch_size


def main():
    directory = "data/train"
    batch_size = 32
    datagen = ImageDataGenerator()
    data_iterator = datagen.flow_from_directory(
        directory,
        target_size=(256, 256),
        class_mode="categorical",
        batch_size=batch_size,
        interpolation="nearest",
    )
    # with Pool(processes=6) as pool:
    #     result = pool.imap(compute_dataset_mean, data_iterator, chunksize=32)
    # mean = batch_size*result
    # print(mean)

    dataset_mean = compute_dataset_mean(data_iterator, 32)
    print(dataset_mean)

    # data_dirs = ["train", "test"]

    # for d in data_dirs:
    #     for sub_d in os.listdir(os.path.join('data', data_dirs[d])):
    #         for file in os.list
    # num_images = 0
    # sum_means = 0.0
    # for d in data_dirs:
    #     for dirpath, dirnames, filenames in os.walk(os.path.join("data", d)):
    #         for f in filenames:
    #             if f.endswith(".jpg"):
    #                 num_images += 1

    #                 img = image.load_img(os.path.join(dirpath, f))
    #                 img_array = image.img_to_array(img)
    #                 mean = tf.math.reduce_mean(img_array)
    #                 # print("mean:", mean)
    #                 sum_means += mean
    #                 # break

    #         # for filename in files:
    #         #     filepath = os.path.join(subdir,filename)

    #         #     if filepath.endswith(".jpg") or filepath.endswith(".png"):
    #         #         print(filepath)
    #     print("num images:", num_images)
    #     # break
    # global_mean = sum_means / num_images
    # print("global mean:", global_mean)


if __name__ == "__main__":
    main()
