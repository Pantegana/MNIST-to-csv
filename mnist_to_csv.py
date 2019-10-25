import numpy as np
import pandas as pd
import struct
import sys

TEST_FILE_NAME = "mnist_test.csv"
TRAIN_FILE_NAME = "mnist_train.csv"

if len(sys.argv) < 5:
    print("You need to provide the names of all (4) the files.")
    sys.exit(1)

#retrieves image data 
def get_img_data(raw_img_data):
    header = struct.unpack('>iiii', raw_img_data[:16])
    num_img = header[1]
    rows = header[2]
    cols = header[3]
    data = struct.unpack('>' + 'B' * rows * cols * num_img, raw_img_data[16:])
    return np.asarray(data).reshape((num_img, rows * cols))

#retrieves labels
def get_label_data(raw_label_data):
    header = struct.unpack('>ii', raw_label_data[:8])
    num_lbl = header[1]
    return np.asarray(struct.unpack('>' + 'B' * num_lbl, raw_label_data[8:]))


def main():
    #open mnist files
    train_img_file = open(sys.argv[1], 'rb')
    train_labels_file = open(sys.argv[2], 'rb')
    test_img_file = open(sys.argv[3], 'rb')
    test_labels_file = open(sys.argv[4], 'rb')
    #read files
    raw_img_train = train_img_file.read()
    raw_labels_train = train_labels_file.read()
    raw_img_test = test_img_file.read()
    raw_labels_test = test_labels_file.read()

    train_img_file.close()
    train_labels_file.close()
    test_img_file.close()
    test_labels_file.close()
    
    #get training the data and put it into a pandas dataframe
    data = get_img_data(raw_img_train)
    labels = get_label_data(raw_labels_train)
    df = pd.DataFrame(data)
    df.insert(loc=0, column='label', value=labels)
    #save as csv
    df.to_csv('mnist_train.csv', sep=',')

    print(f"Test data has been saved to {TEST_FILE_NAME}.")

    #get test the data and put it into a pandas dataframe
    data = get_img_data(raw_img_train)
    labels = get_label_data(raw_labels_train)
    df = pd.DataFrame(data)
    df.insert(loc=0, column='label', value=labels)
    #save as csv
    df.to_csv('mnist_test.csv', sep=',')

    print(f"Training data has been saved to {TRAIN_FILE_NAME}.")

if __name__ == "__main__":
    main()
