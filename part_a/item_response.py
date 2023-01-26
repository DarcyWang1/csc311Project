from matplotlib import pyplot as plt
from scipy import sparse
from scipy.sparse import csr_matrix

from utils import *

import numpy as np


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    tb = (np.asarray([theta]).T @ (np.asarray([[1]] * beta.shape[0])).T) - \
            (np.asarray([[1]] * theta.shape[0]) @ np.asarray([beta]))

    y = sigmoid(tb)

    # d = np.nan_to_num(csr_matrix.toarray(data), nan=-1)
    # msk1 = d > 0
    # mask2 = d >= 0
    # log_lklihood = np.sum((msk1 * tb)) - np.sum(mask2 * (np.log(1 + np.e ** tb)))

    log_lklihood = np.sum(np.nan_to_num(csr_matrix.toarray(data) * np.log(y) +
                    (1 - csr_matrix.toarray(data)) * (np.log(1 - y)), nan=0))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################

    # d = np.nan_to_num(csr_matrix.toarray(data), nan=0)
    # msk1 = d > 0
    # mask2 = np.nan_to_num(csr_matrix.toarray(data), nan=-1) >= 0

    tb = (np.asarray([theta]).T @ (np.asarray([[1]] * beta.shape[0])).T) - \
            (np.asarray([[1]] * theta.shape[0]) @ np.asarray([beta]))
    y = sigmoid(tb)
    a = csr_matrix.toarray(data)

    # c=mask2*sigmoid(tb)
    # theta = theta - lr * (np.sum(d, axis=1) - np.sum(mask2*sigmoid(tb), axis=1))

    # test = test_gredient(data,theta,beta)
    # caulated = np.sum(np.nan_to_num((a/y-(1-a)/(1-y)) * (y-1) * y,nan=0),axis=1)
    # a = caulated/(np.sum(caulated**2)**0.5)-test/(np.sum(test**2)**0.5)
    # print(a)

    theta = theta - lr * np.sum(np.nan_to_num((a / y - (1 - a) / (1 - y)) * (y - 1) * y, nan=0), axis=1)
    tb = (np.asarray([theta]).T @ (np.asarray([[1]] * beta.shape[0])).T) - \
            (np.asarray([[1]] * theta.shape[0]) @ np.asarray([beta]))
    y = sigmoid(tb)
    beta = beta + lr * np.sum(np.nan_to_num((a / y - (1 - a) / (1 - y)) * (y - 1) * y, nan=0), axis=0)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta

'''
def test_gredient(data, theta, beta):
    tb = (np.asarray([theta]).T @ (np.asarray([[1]] * beta.shape[0])).T) - \
            (np.asarray([[1]] * theta.shape[0]) @ np.asarray([beta]))
    y = sigmoid(tb)
    a = csr_matrix.toarray(data)

    # c=mask2*sigmoid(tb)
    # theta = theta - lr * (np.sum(d, axis=1) - np.sum(mask2*sigmoid(tb), axis=1))
    # test = test_gredient(data, theta, beta)
    caulated = np.sum(
        np.nan_to_num((a / y - (1 - a) / (1 - y)) * (1 - y) * y, nan=0), axis=1)
    caulated = caulated / (np.sum(caulated ** 2) ** 0.5)

    gt = np.asarray([0] * theta.shape[0])
    ide = (0.01 * np.identity(theta.shape[0]))
    # gb =np.zeros(beta)
    for i in range(len(gt)):
        # a = np.asarray([0.0]*theta.shape[0])
        # a[i]=0.01

        gt[i] = neg_log_likelihood(data, theta + ide[i], beta) - neg_log_likelihood(data,theta - ide[i], beta)
        if gt[i] != 0:
            print(gt[i], caulated[i])
    return gt

'''
def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    t, b = data.shape
    theta = np.full((t,), 0)
    beta = np.full((b,), 0)

    val_acc_lst = []
    tra_log_like=[]
    log_like_list=[]

    vd = fix_valdation(val_data)

    for _ in range(iterations):

        # Record progress
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        tra_log_like.append(neg_lld)
        log_like_list.append(neg_log_likelihood(vd, theta=theta, beta=beta))
        #print("NLLK: {} \t Score: {}".format(neg_lld, score))

        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, log_like_list,tra_log_like


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def fix_valdation(valdation):
    matrix = np.full((max(valdation['user_id'])+1,max(valdation['question_id'])+1),np.nan)
    for i in range(len(valdation['user_id'])):
        matrix[valdation['user_id'][i],valdation['question_id'][i]] = valdation['is_correct'][i]
    return sparse.csr_matrix(matrix)
    #print(valdation.keys())
def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################

    # Hyperparameters
    lr = 0.01
    iter_num = 10

    theta, beta, log_like_list,tra_log_like = irt(sparse_matrix, val_data, lr, iter_num)

    plt.plot(range(iter_num), log_like_list, color='purple', label='validation')
    plt.plot(range(iter_num), tra_log_like, color='blue',label='train')
    #save_plot("IRT Validation Accuracy Against Iter", 'Iteration', 'Validation Accuracy')
    save_plot("IRT neg log loss Against Iter", 'Iteration',
              'neg log loss')

    test_acc = evaluate(test_data, theta, beta)
    print(f"Test Accuracy: {test_acc}")

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################

    js = [1, 50, 500]

    colors = ['blue', 'purple', 'green']
    theta.sort()
    for i in range(len(js)):
        plt.plot(theta, sigmoid(theta - beta[js[i]]), color=colors[i], label=f"j={js[i]}")
    save_plot("IRT Prob Against Theta", 'Theta', 'P(cij = 1)')

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
