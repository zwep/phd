import os
os.environ["OMP_NUM_THREADS"] = "10" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "10" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "10" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "10" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "10" # export NUMEXPR_NUM_THREADS=6
import sys
import itertools
import multiprocessing as mp
import json
import os
from IPython.display import display, clear_output, HTML
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

username = os.environ.get('USER', os.environ.get('USERNAME'))
remote = False
if username != 'bugger':
    remote = True

mia_path_code = '/home/bugger/PycharmProjects/8dc00-mia-private/code'
mia_path_data = '/home/bugger/PycharmProjects/8dc00-mia-private/data'
if remote:
    mia_path_code = '/data/seb/code/8dc00-mia-private/code'
    mia_path_data = '/data/seb/code/8dc00-mia-private/data'

sys.path.append(mia_path_data)
sys.path.append(mia_path_code)
import cad_util as util

"""
Redefine the training class...
"""


class Training:
    def __init__(self, data_path, n_hidden_features=None,
                 n_epochs=None, batchsize=None, learing_rate=None, p_reduce=None):
        self.data_path = data_path
        self.batchsize = batchsize
        self.n_epochs = n_epochs
        self.learning_rate = learing_rate
        self.n_hidden_features = n_hidden_features
        self.p_reduce = p_reduce
        self.data_preprocessing()
        self.define_shapes()

    def data_preprocessing(self):
        print('Preprocessing data')
        ## load dataset (images and labels y)
        # fn = '../data/nuclei_data_classification.mat'
        mat = scipy.io.loadmat(self.data_path)
        print('\tData loaded')
        if self.p_reduce is None:
            self.p_reduce = 1

        training_images = mat["training_images"]  # (24, 24, 3, 14607)
        n_train = training_images.shape[-1]
        sel_n_train = int(n_train * self.p_reduce)
        training_images = training_images[:, :, :, :sel_n_train]
        self.training_y = mat["training_y"]  # (14607, 1)
        self.training_y = self.training_y[:sel_n_train]

        validation_images = mat["validation_images"]  # (24, 24, 3, 7303)
        n_val = validation_images.shape[-1]
        sel_n_val = int(n_val * self.p_reduce)
        validation_images = validation_images[:, :, :, :sel_n_val]
        self.validation_y = mat["validation_y"]  # (7303, 1)
        self.validation_y = self.validation_y[:sel_n_val]

        test_images = mat["test_images"]  # (24, 24, 3, 20730)
        n_test = test_images.shape[-1]
        sel_n_test = int(n_test * self.p_reduce)
        test_images = test_images[:, :, :, :sel_n_test]
        self.test_y = mat["test_y"]  # (7303, 1)
        self.test_y = self.test_y[:sel_n_test]

        ## dataset preparation
        # Reshape matrices and normalize pixel values
        self.training_x, self.validation_x, self.test_x = util.reshape_and_normalize(training_images,
                                                                                     validation_images,
                                                                                     test_images)
        print('\tData preprocessed')
        # Visualize several training images classified as large or small
        # util.visualize_big_small_images(self.training_x, self.training_y, training_images.shape)
        #util.visualize_big_small_images(training_images, self.training_y)

    def define_shapes(self):
        print("Defining shapes")
        if self.learning_rate is None:
            self.learning_rate = 0.001
        if self.batchsize is None:
            self.batchsize = 128
        if self.n_hidden_features is None:
            n_hidden_features = 1000

        in_features = self.training_x.shape[1]
        out_features = 1  # Classification problem, so you want to obtain 1 value (a probability) per image

        # Define shapes of the weight matrices
        # ---------------------------------------------------------------------#
        # TODO: Create two variables: w1_shape and w2_shape, and define them as
        # follows (as a function of variables defined above)
        # self.w1_shape = (.. , ..)
        # self.w2_shape = (.. , ..)
        # !studentstart
        self.w1_shape = (in_features, n_hidden_features)
        self.w2_shape = (n_hidden_features, out_features)
        # !studentend
        # ---------------------------------------------------------------------#
        print("\t  Done defining shapes")
        return {'w1_shape': self.w1_shape,
                'w2_shape': self.w2_shape}

    def launch_training(self):
        # Define empty lists for saving training progress variables
        training_loss = []
        validation_loss = []
        Acc = []
        steps = []

        # randomly initialize model weights
        self.weights = util.init_model(self.w1_shape, self.w2_shape)

        # print('> Start training ...')
        # Train for n_epochs epochs
        if self.n_epochs is None:
            self.n_epochs = 100

        # print()
        for epoch in range(self.n_epochs):
            #print(f'Epoch {epoch} / {self.n_epochs}', end='\r')
            # Shuffle training images every epoch
            training_x, training_y = util.shuffle_training_x(self.training_x, self.training_y)

            n_batches = self.training_x.shape[0] // self.batchsize
            # n_batches = 10
            for batch_i in range(n_batches):
                # print("Batch ", batch_i, " / ", n_batches, end="\r")
                ## sample images from this batch
                batch_x = training_x[self.batchsize * batch_i: self.batchsize * (batch_i + 1)]
                batch_y = training_y[self.batchsize * batch_i: self.batchsize * (batch_i + 1)]

                ## train on one batch
                # Forward pass
                hidden, output = self.forward(batch_x, self.weights)
                # Backward pass
                self.weights = self.backward(batch_x, batch_y, output, hidden, self.weights)

                ## Save values of loss function for plot
                training_loss.append(util.loss(output, batch_y))
                steps.append(epoch + batch_i / (self.training_x.shape[0] // self.batchsize))

            ## Validation images trhough network
            # Forward pass only (no backward pass in inference phase!)
            _, val_output = self.forward(self.validation_x, self.weights)
            # Save validation loss
            val_loss = util.loss(val_output, self.validation_y)
            validation_loss.append(val_loss)
            accuracy = (self.validation_y == np.round(val_output)).sum() / (self.validation_y.shape[0])
            Acc.append(accuracy)

            # Plot loss function and accuracy of validation set
            # clear_output(wait=True)
            # fig, ax = plt.subplots(1, 2, figsize=(15, 5))
            # ax[0].plot(steps, training_loss)
            # ax[0].plot(range(1, len(validation_loss) + 1), validation_loss, '.')
            # ax[0].legend(['Training loss', 'Validation loss'])
            # ax[0].set_title(f'Loss curves after {epoch + 1}/{self.n_epochs} epochs')
            # ax[0].set_ylabel('Loss')
            # ax[0].set_xlabel('epochs')
            # ax[0].set_xlim([0, 100])
            # ax[0].set_ylim([0, max(training_loss)])
            # ax[1].plot(Acc)
            # ax[1].set_title(f'Validation accuracy after {epoch + 1}/{self.n_epochs} epochs')
            # ax[1].set_ylabel('Accuracy')
            # ax[1].set_xlabel('epochs')
            # ax[1].set_xlim([0, 100])
            # ax[1].set_ylim([min(Acc), 0.8])
            # plt.show()
        # print('> Training finished')
        return training_loss, validation_loss, Acc

    def pass_on_test_set(self):
        # Forward pass on test set
        _, test_output = self.forward(self.test_x, self.weights)
        test_accuracy = (self.test_y == np.round(test_output)).sum() / (self.test_y.shape[0])
        # print('Test accuracy: {:.2f}'.format(test_accuracy))

        # Plot final test predictions
        large_list = test_output[self.test_y == 1]
        small_list = test_output[self.test_y == 0]
        # plt.figure()
        # plt.hist(small_list, 50, alpha=0.5)
        # plt.hist(large_list, 50, alpha=0.5)
        # plt.legend(['Small (label = 0)', 'Large (label = 1)'], loc='upper center')
        # plt.xlabel('Prediction')
        # plt.title('Final test set predictions')
        # plt.show()
        return test_accuracy

    def forward(self, x, weights):
        w1 = weights['w1']
        w2 = weights['w2']

        hidden = util.sigmoid(np.dot(x, w1))
        output = util.sigmoid(np.dot(hidden, w2))

        return hidden, output

    def backward(self, x, y, output, hidden, weights):
        w1 = weights['w1']
        w2 = weights['w2']

        # Caluclate the derivative with the use of the chain rule
        dL_dw2 = np.dot(hidden.T, (2 * (output - y) * util.sigmoid_derivative(output)))
        dL_dw1 = np.dot(x.T, (
                    np.dot(2 * (output - y) * util.sigmoid_derivative(output), w2.T) * util.sigmoid_derivative(hidden)))

        # update the weights with the derivative (slope) of the loss function
        # ---------------------------------------------------------------------#
        # TODO: Update the variables: w1 and w2, and define them as
        # follows (as a function of learning_rate, dL_dw1, and dL_dw2)
        # w1 = w1 - ...
        # w2 = w2 - ...
        # !studentstart
        w1 = w1 - self.learning_rate * dL_dw1
        w2 = w2 - self.learning_rate * dL_dw2
        # !studentend
        # ---------------------------------------------------------------------#
        return {'w1': w1,
                'w2': w2}


def get_stuff(i_options):
    i_epoch, i_batch_size, i_learning_rate = i_options
    training_obj = Training(data_path=data_path, batchsize=i_batch_size,
                            n_epochs=i_epoch, learing_rate=i_learning_rate,
                            p_reduce=0.8)
    training_loss, validation_loss, Acc = training_obj.launch_training()
    test_loss = training_obj.pass_on_test_set()
    temp_dict = {'learning_rate': i_learning_rate,
                 'batch_size': i_batch_size,
                 'epoch': i_epoch,
                 'acc': Acc,
                 'val_loss': validation_loss,
                 'train_loss': training_loss,
                 'test_loss': test_loss}
    return temp_dict


if __name__ == "__main__":
    data_path = '/home/bugger/PycharmProjects/8dc00-mia-private/data/nuclei_data_classification.mat'
    ddest_a = '/home/bugger/Documents/TUE/AssistentDocent/Assignment2/task_5_a.json'
    ddest_b = '/home/bugger/Documents/TUE/AssistentDocent/Assignment2/task_5_b.json'
    if remote:
        data_path = '/data/seb/code/8dc00-mia-private/data/nuclei_data_classification.mat'
        ddest_a = '/data/seb/course_8dc00/task_5_a.json'
        ddest_b = '/data/seb/course_8dc00/task_5_b.json'

    # data_path, n_epochs = None, batchsize = None, learing_rate = None):
    n_epoch_list = list(range(50, 500, 50))
    batch_size_list = list(range(500, 1000, 100))
    learning_rate_list = list(np.arange(1e-4, 1e-3, 5e-4))

    all_prod = itertools.product(n_epoch_list, batch_size_list, learning_rate_list)
    N = mp.cpu_count()
    N_sel = N // 4
    with mp.Pool(N_sel) as p:
        result_list = p.map(get_stuff, all_prod)

    json_ser = ", ".join([json.dumps(x) for x in result_list])
    with open(ddest_a, 'a') as f:
        f.write(json_ser)

    sel_batch_size = 10
    sel_learning_rate = 0.0001
    hidden_feature_list = [500, 1000, 2000]
    sel_epoch = 200
    result_list_hidden = []
    for i_hidden_feature in hidden_feature_list:
        training_obj = Training(data_path=data_path, batchsize=sel_batch_size,
                                n_epochs=sel_epoch, learing_rate=sel_learning_rate,
                                n_hidden_features=i_hidden_feature)
        training_loss, validation_loss, Acc = training_obj.launch_training()
        test_loss = training_obj.pass_on_test_set()
        temp_dict = {'n_hidden': i_hidden_feature,
                     'acc': Acc,
                     'val_loss': validation_loss,
                     'train_loss': training_loss,
                     'test_loss': test_loss}
        result_list_hidden.append(temp_dict)

    json_ser = ", ".join([json.dumps(x) for x in result_list])
    with open(ddest_b, 'a') as f:
        f.write(json_ser)