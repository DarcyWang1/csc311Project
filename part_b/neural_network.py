from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch

import matplotlib.pyplot as plt

import random
layer = 4

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

    train_matrix = np.append(train_matrix, students_subject_accuracy, axis = 1)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return num_questions, zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_features, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_features, k)
        self.q = nn.Linear(k, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################

        sigmoid_fn = nn.Sigmoid()
        g_out = sigmoid_fn(self.g(inputs))
        q1_out = sigmoid_fn(self.q(g_out))
        # q2_out = sigmoid_fn(self.q(q1_out))
        # q3_out = sigmoid_fn(self.q(q2_out))
        h_out = sigmoid_fn(self.h(q1_out))
        out = h_out
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_questions, num_epoch, global_max_acc):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function.

    # Record Progress
    epochs, train_losses, valid_accs = [], [], []
    max_acc = -1

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    for epoch in range(0, num_epoch):
        train_loss = 0.

        loss = 0
        count = 0

        student_indices = list(range(num_student))
        random.shuffle(student_indices)

        for user_id in student_indices:
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()[:, :num_questions]

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())[:, :num_questions]
            target[0][nan_mask] = output[0][nan_mask]

            # L2 regularizer
            loss += torch.sum((output - target) ** 2.) + (lamb / 2) * model.get_weight_norm()
            count += 1
            if count % 8 == 0 or count == num_student - 1:

                loss.backward()

                train_loss += loss.item()
                optimizer.step()

                loss = 0

        valid_acc = evaluate(model, zero_train_data, valid_data)

        # Record Progress
        if valid_acc > max_acc:
            max_acc = valid_acc
            if valid_acc > global_max_acc:
                torch.save(model.state_dict(), "model.pth")

        epochs.append(epoch)
        train_losses.append(train_loss)
        valid_accs.append(valid_acc)

        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}\t Max Acc: {}".format(epoch, train_loss, valid_acc, max_acc))

    return epochs, train_losses, valid_accs, max_acc

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def plot_against_epochs(title, ylabel, param_name, params_list, epochs_list, train_metrics_list):
    colors = ['blue', 'purple', 'green', 'red', 'cyan']

    for i in range(len(epochs_list)):
        plt.plot(epochs_list[i], train_metrics_list[i], color=colors[i], label=f"{param_name}={params_list[i]}")

    plt.ylabel(f'{ylabel}')
    plt.xlabel('Epoch')
    plt.title(title)
    plt.legend()

    plt.savefig(f'{title}.png')
    plt.clf()


def main():

    random.seed(0)
    torch.manual_seed(0)

    num_questions, zero_train_matrix, train_matrix, valid_data, test_data = load_data()
    num_features = zero_train_matrix.shape[1]
    global_max_acc, global_max_acc_k = 0, 0

    epochs_list, train_losses_list, valid_accs_list, max_accs_list = [], [], [], []
    model = None
    # Set optimization hyperparameters.
    lr = 0.01
    num_epoch = 300
    lamb = 0

    # Set model hyperparameters.
    k_list = [10, 50, 100, 200, 500]
    for k in k_list:
        model = AutoEncoder(num_features, num_questions, k)

        print(f"Start Training with k={k} lamb={lamb} lr={lr}")
        epochs, train_losses, valid_accs, max_acc = train(model, lr, lamb, train_matrix,
            zero_train_matrix, valid_data, num_questions, num_epoch, global_max_acc)

        epochs_list.append(epochs)
        train_losses_list.append(train_losses)
        valid_accs_list.append(valid_accs)
        max_accs_list.append(max_acc)
        if max_acc > global_max_acc:
            global_max_acc = max_acc
            global_max_acc_k = k

    plot_against_epochs(f"Training Loss against Epoch lr={lr}", "Training Loss",
        "k", k_list, epochs_list, train_losses_list)
    plot_against_epochs(f"Validation Accuracy against Epoch lr={lr}", "Validation Accuracy",
        "k", k_list, epochs_list, valid_accs_list)

    for i in range(len(k_list)):
        print(f"k={k_list[i]}: Max Validation Accuracy: {max_accs_list[i]}")

    print(f"Loading model with k={global_max_acc_k} and validation accuracy: {global_max_acc}")
    model = AutoEncoder(num_features, num_questions, global_max_acc_k)
    model.load_state_dict(torch.load("model.pth"))

    test_acc = evaluate(model, zero_train_matrix, test_data)
    print(f"Test Accuracy: {test_acc}")

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def experement():
    random.seed(0)
    torch.manual_seed(0)


    global_max_acc, global_max_acc_k = 0, 0

    epochs_list, train_losses_list, valid_accs_list, max_accs_list = [], [], [], []
    model = None
    # Set optimization hyperparameters.
    lr = 0.01
    num_epoch = 200
    lamb = 0.01

    # Set model hyperparameters.
    #k_list = [10, 50, 100, 200, 500]
    k=200
    #for k in k_list:
    t_list=[]
    global layer
    layer = 4
    for i in [0.1,0.05,0.01,0]:

        num_questions, zero_train_matrix, train_matrix, valid_data, test_data = load_data()
        num_features = zero_train_matrix.shape[1]
        model = AutoEncoder(num_features, num_questions, k)

        #print(f"Start Training with k={k} lamb={lamb} lr={lr}")
        epochs, train_losses, valid_accs, max_acc = train(model, lr, i,
                                                          train_matrix,
                                                          zero_train_matrix,
                                                          valid_data,
                                                          num_questions,
                                                          num_epoch,
                                                          global_max_acc)

        epochs_list.append(epochs)
        train_losses_list.append(train_losses)
        valid_accs_list.append(valid_accs)
        max_accs_list.append(max_acc)
        if max_acc > global_max_acc:
            global_max_acc = max_acc
            global_max_acc_k = k
        model1 = AutoEncoder(num_features, num_questions, global_max_acc_k)
        model1.load_state_dict(torch.load("model.pth"))

        test_acc = evaluate(model1, zero_train_matrix, test_data)
        print(test_acc)
        t_list.append(test_acc)
    plot_against_epochs(f"Training Loss against Epoch for each lambda", "Training Loss",
                        "lambda", [0.1,0.05,0.01,0], epochs_list, train_losses_list)
    plot_against_epochs(f"Validation Accuracy against Epoch for each lambda",
                        "Validation Accuracy",
                        "lambda", [0.1,0.05,0.01,0], epochs_list, valid_accs_list)

    for i in range(len([0.1,0.05,0.01,0])):
        print(f"k={[0.1,0.05,0.01,0][i]}: Max Validation Accuracy: {max_accs_list[i]}")

    print(
        f"Loading model with k={global_max_acc_k} and validation accuracy: {global_max_acc}")
    model = AutoEncoder(num_features, num_questions, global_max_acc_k)
    model.load_state_dict(torch.load("model.pth"))

    test_acc = evaluate(model, zero_train_matrix, test_data)
    print(f"Test Accuracy: {test_acc}")
    print(t_list)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
    #experement()
