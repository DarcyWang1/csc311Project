from matplotlib import pyplot as plt
from scipy import sparse
from scipy.sparse import csr_matrix

from part_a.utils import save_plot
from utils import *

import numpy as np
from scipy.sparse import csr_matrix


layer=4

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
    print('a')
    for _ in range(iterations):
        theta, beta = update_theta_beta(data, lr, theta, beta)

        # Record progress
        #neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        #score = evaluate(data=val_data, theta=theta, beta=beta)
        #val_acc_lst.append(score)
        #print("NLLK: {} \t Score: {}".format(neg_lld, score))

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst


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

    theta, beta, val_acc_lst = irt(sparse_matrix, val_data, lr, iter_num)

    plt.plot(range(iter_num), val_acc_lst, color='purple')
    save_plot("IRT Validation Accuracy Against Iter", 'Iteration', 'Validation Accuracy')

    test_acc = evaluate(test_data, theta, beta)
    print(f"Test Accuracy: {test_acc}")

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################

    js = [1, 50, 100]

    colors = ['blue', 'purple', 'green']
    for i in range(len(js)):
        plt.plot(theta, sigmoid(theta - beta[js[i]]), 'o', color=colors[i], label=f"j={js[i]}", markersize=1)
    save_plot("IRT Prob Against Theta", 'Theta', 'P(cij = 1)')

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


















def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    train_data = load_train_csv(base_path)
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)
    student_data = load_student_meta(base_path)
    question_data = load_question_meta(base_path)

    num_student, num_questions = train_matrix.shape

    min_age, max_age = 100, -1
    for user_id in student_data:
        age = student_data[user_id][1]
        if age:
            if age < min_age:
                min_age = age
            if age > max_age:
                max_age = age

    age_sum = 0
    age_count = 0
    premium_pupil_sum = 0
    premium_pupil_count = 0

    for user_id in student_data:
        age = student_data[user_id][1]
        premium_pupil = student_data[user_id][2]
        if age:
            student_data[user_id][1] = (age - min_age) / (max_age - min_age)
            age_sum = student_data[user_id][1]
            age_count += 1
        if premium_pupil:
            premium_pupil_sum += premium_pupil
            premium_pupil_count += 1

    premium_pupil_avg = premium_pupil_sum / premium_pupil_count
    age_avg = age_sum / age_count


    student_subject_score_table = {}
    num_age_groups = 5
    age_group_subject_score_table = []
    gender_age_group_subject_score_table = []
    for _ in range(num_age_groups):
        age_group_subject_score_table.append({})
    for _ in range(num_age_groups * 2):
        gender_age_group_subject_score_table.append({})

    students = []
    for user_id in range(num_student):
        if not student_data[user_id][1]:
            student_data[user_id][1] = age_avg
        if not student_data[user_id][2]:
            student_data[user_id][2] = premium_pupil_avg
        students.append(student_data[user_id].copy())


    students = np.array(students)
    # train_matrix = np.append(train_matrix, students, axis = 1)

    user_age_interval = 1 / num_age_groups

    subject_score_table = {}

    for i in range(len(train_data['user_id'])):
        user_id = train_data['user_id'][i]
        question_id = train_data['question_id'][i]
        is_correct = train_data['is_correct'][i]

        user_gender = student_data[user_id][0]
        user_age = student_data[user_id][1]
        question_subjects = question_data[question_id]

        if user_id not in student_subject_score_table:
            student_subject_score_table[user_id] = {}


        age_group_idx = int(user_age / user_age_interval)
        if age_group_idx == num_age_groups:
            age_group_idx = num_age_groups - 1

        for question_subject in question_subjects:

            if question_subject not in student_subject_score_table[user_id]:
                student_subject_score_table[user_id][question_subject] = {"count": 0, "num_correct": 0, "accuracy": 0}

            student_subject_score_table[user_id][question_subject]["count"] += 1

            if is_correct:
                student_subject_score_table[user_id][question_subject]["num_correct"] += 1

            student_subject_score_table[user_id][question_subject]["accuracy"] = student_subject_score_table[user_id][question_subject]["num_correct"] / student_subject_score_table[user_id][question_subject]["count"]

            age_group_dict = age_group_subject_score_table[age_group_idx]

            if question_subject not in age_group_dict:
                age_group_dict[question_subject] = {"count": 0, "num_correct": 0, "accuracy": 0}

            age_group_dict[question_subject]["count"] += 1

            if is_correct:
                age_group_dict[question_subject]["num_correct"] += 1
            age_group_dict[question_subject]["accuracy"] = age_group_dict[question_subject]["num_correct"] / age_group_dict[question_subject]["count"]

            if question_subject not in subject_score_table:
                subject_score_table[question_subject] = {"count": 0, "num_correct": 0, "accuracy": 0}

            subject_score_table[question_subject]["count"] += 1

            if is_correct:
                subject_score_table[question_subject]["num_correct"] += 1
            subject_score_table[question_subject]["accuracy"] = subject_score_table[question_subject]["num_correct"] / subject_score_table[question_subject]["count"]

            if user_gender == 1 or user_gender == 2:
                if user_gender == 1:
                    gender_age_group_dict = gender_age_group_subject_score_table[age_group_idx]
                if user_gender == 2:
                    gender_age_group_dict = gender_age_group_subject_score_table[age_group_idx + num_age_groups]

                if question_subject not in gender_age_group_dict:
                    gender_age_group_dict[question_subject] = {"count": 0, "num_correct": 0, "accuracy": 0}

                gender_age_group_dict[question_subject]["count"] += 1

                if is_correct:
                    gender_age_group_dict[question_subject]["num_correct"] += 1
                gender_age_group_dict[question_subject]["accuracy"] = gender_age_group_dict[question_subject]["num_correct"] / gender_age_group_dict[question_subject]["count"]


    students_subject_accuracy = []
    for user_id in range(num_student):
        student_subject_accuracy = []

        user_gender = student_data[user_id][0]
        user_age = student_data[user_id][1]
        age_group_idx = int(user_age / user_age_interval)
        if age_group_idx == num_age_groups:
            age_group_idx = num_age_groups - 1

        for subject_id in subject_score_table:
            if layer>3:
#                print('4')
                if subject_id in student_subject_score_table[user_id] and student_subject_score_table[user_id][subject_id]["count"] > 5:
                    student_subject_accuracy.append(student_subject_score_table[user_id][subject_id]["accuracy"])
                    continue
            if layer>2:
    #            print('3')
                if user_gender == 1 or user_gender == 2:
                    if user_gender == 1:
                        gender_age_group_dict = gender_age_group_subject_score_table[age_group_idx]
                    if user_gender == 2:
                        gender_age_group_dict = gender_age_group_subject_score_table[age_group_idx + num_age_groups]

                    if subject_id in gender_age_group_dict:
                        student_subject_accuracy.append(gender_age_group_dict[subject_id]["accuracy"])
                        assert(gender_age_group_dict[subject_id]["count"] > 0)
                        continue
            if layer>1:
#                print(2)
                if subject_id in age_group_subject_score_table[age_group_idx]:
                    student_subject_accuracy.append(age_group_subject_score_table[age_group_idx][subject_id]["accuracy"])
                    assert(age_group_subject_score_table[age_group_idx][subject_id]["count"] > 0)
                    continue
#            print(1)
            student_subject_accuracy.append(subject_score_table[subject_id]["accuracy"])
            assert(subject_score_table[subject_id]["count"] > 0)

        students_subject_accuracy.append(student_subject_accuracy)
    s = max(((max((list(question_data.values())),key=lambda l:max(l)))))
    q = np.zeros((s+1,max(question_data.keys())+1))
    for i in question_data.keys():
        for j in question_data[i]:
            q[j][i]=1
    #print(question_data)
    result = np.moveaxis(np.full((max(student_data.keys())+1,)+q.shape,q),0,1)
    result[result<1]=np.nan

    #print(np.moveaxis(result,0,1).shape)
    return result




def update_theta_beta2(data, lr, theta, beta):
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
    D,M = theta.shape
    N,MM = beta.shape
    assert M==MM
    t = np.full((N,D,M),theta)
    b = np.full((D,N,M),beta)
    b = np.moveaxis(b,0,1)
    tb = np.sum(t-b,axis=-1)
    tb = tb.T


    #tb = (np.asarray([theta]).T @ (np.asarray([[1]] * beta.shape[0])).T) - \
     #       (np.asarray([[1]] * theta.shape[0]) @ np.asarray([beta]))
    y = sigmoid(tb)
    #a = csr_matrix.toarray(data)

    # c=mask2*sigmoid(tb)
    # theta = theta - lr * (np.sum(d, axis=1) - np.sum(mask2*sigmoid(tb), axis=1))

    # test = test_gredient(data,theta,beta)
    # caulated = np.sum(np.nan_to_num((a/y-(1-a)/(1-y)) * (y-1) * y,nan=0),axis=1)
    # a = caulated/(np.sum(caulated**2)**0.5)-test/(np.sum(test**2)**0.5)
    # print(a)
    print(np.nan_to_num((a / y - (1 - a) / (1 - y)) * (y - 1) * y, nan=0).shape)
    theta = theta - lr * np.sum(np.nan_to_num((a / y - (1 - a) / (1 - y)) * (y - 1) * y, nan=0), axis=-1).T
    #tb = (np.asarray([theta]).T @ (np.asarray([[1]] * beta.shape[0])).T) - \
     #       (np.asarray([[1]] * theta.shape[0]) @ np.asarray([beta]))
    t = np.full((N, D, M), theta)
    b = np.full((D, N, M), beta)
    b = np.moveaxis(b, 0, 1)
    tb = np.sum(t - b, axis=-1)
    tb = tb.T
    y = sigmoid(tb)
    beta = beta + lr * np.sum(np.nan_to_num((a / y - (1 - a) / (1 - y)) * (y - 1) * y, nan=0), axis=-2).T

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta





def irt2(data, val_data, lr, iterations):
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
    m,t, b = data.shape
    theta = np.full((t,m), 0)
    beta = np.full((b,m), 0)

    val_acc_lst = []
    print('a')
    for _ in range(iterations):
        print(_)
        theta, beta = update_theta_beta2(data, lr, theta, beta)

        # Record progress
        #neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        #score = evaluate(data=val_data, theta=theta, beta=beta)
        #val_acc_lst.append(score)
        #print("NLLK: {} \t Score: {}".format(neg_lld, score))

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst







def data2MMatrex(data):
    matrix = np.full(
        (max(data['user_id']) + 1, max(data['question_id']) + 1),
        np.nan)
    for i in range(len(data['user_id'])):
        matrix[data['user_id'][i], data['question_id'][i]] = \
        data['is_correct'][i]
    return matrix




if __name__ == "__main__":
    #main()


    a=np.asarray(load_data())
    b = csr_matrix.toarray(load_train_sparse("../data"))
    test_data = load_public_test_csv("../data")
    print(a.shape)
    print(b.shape)
    c = a*np.full(a.shape,(b))
    aa,s,t = c.shape
    buffer = np.zeros((s,t))
    count=np.zeros((s,t))
    val_data = load_valid_csv("../data")
    lr=0.01
    iter_num = 10
    theta, beta, val_acc_lst = irt2(c, val_data, lr, iter_num)

    plt.plot(range(iter_num), val_acc_lst, color='purple')
    save_plot("IRT Validation Accuracy Against Iter", 'Iteration',
              'Validation Accuracy')

    test_acc = evaluate(test_data, theta, beta)
    print(f"Test Accuracy: {test_acc}")
    '''
    for i in range(c.shape[0]):
        print(f"{i}/{c.shape[0]}")
        #print(c[i].shape)
        msk = np.full(c[i].shape,np.nan)
        msk[a[i]==1]=1
        irt_matrix = sparse.csr_matrix(c[i])
        theta, beta, _ = irt(irt_matrix, val_data, lr, iter_num)
        tb = (np.asarray([theta]).T @ (np.asarray([[1]] * beta.shape[0])).T) - \
             (np.asarray([[1]] * theta.shape[0]) @ np.asarray([beta]))
        #print(sigmoid(tb).shape)
        ##print(msk.shape)
        #print((msk*sigmoid(tb)).shape)
        buffer+= np.nan_to_num(msk*sigmoid(tb),nan=0)
        count+=np.nan_to_num(msk*sigmoid(tb),nan=-1)>=0
    result = buffer/count
    print(result)
    prediction = (buffer) > 0.5
    val_matrex = data2MMatrex(val_data)
    test_matrex = data2MMatrex(test_data)
    test_acc = np.sum(
        np.nan_to_num(1 - np.abs(prediction - test_matrex), nan=0)) / np.sum(
        np.nan_to_num(test_matrex > -1, nan=0))
    val_acc = np.sum(
        np.nan_to_num(1 - np.abs(prediction - val_matrex), nan=0)) / np.sum(
        np.nan_to_num(val_matrex > -1, nan=0))

    print(val_acc)
    print(test_acc)
    #print(b)
'''
