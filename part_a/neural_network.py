from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch

import matplotlib.pyplot as plt

torch.manual_seed(0)

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
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
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

        activate_fn = nn.Sigmoid()
        g_out = activate_fn(self.g(inputs))
        h_out = activate_fn(self.h(g_out))
        out = h_out
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch, global_max_acc):
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
    epochs, train_losses, valid_accs, valid_losses = [], [], [], []
    max_acc = -1

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.)
            # L2 regularizer
            if lamb and lamb > 0:
                loss += (lamb / 2) * model.get_weight_norm()

            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc, valid_loss = evaluate(model, zero_train_data, valid_data)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))

        # Record Progress
        if valid_acc > max_acc:
            max_acc = valid_acc
            if valid_acc > global_max_acc:
                torch.save(model.state_dict(), "model.pth")
        epochs.append(epoch)
        train_losses.append(train_loss)
        valid_accs.append(valid_acc)
        valid_losses.append(valid_loss)

    return epochs, train_losses, valid_accs, valid_losses, max_acc

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
    loss = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        output_item = output[0][valid_data["question_id"][i]].item()
        guess = output_item >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    
        target = 1 if valid_data["is_correct"][i] else 0
        loss += (output_item - target) ** 2.

    return correct / float(total), loss


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
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################

    torch.manual_seed(0)

    epochs_list, train_losses_list, valid_accs_list, valid_losses_list, max_accs_list = [], [], [], [], []
    global_max_acc, global_max_acc_k = 0, 0
    model = None

    # Set optimization hyperparameters.
    lr = 0.01
    num_epoch = 100
    # lamb_list = [0.001, 0.01, 0.1, 1]
    lamb_list = [0]
    num_questions = zero_train_matrix.shape[1]

    # Set model hyperparameters.
    k_list = [10, 50, 100, 200, 500]
    # k_list = [50]
    for k in k_list:
        for lamb in lamb_list:
            model = AutoEncoder(num_questions, k)
            
            epochs, train_losses, valid_accs, valid_losses, max_acc = train(model, lr, lamb, train_matrix,
                zero_train_matrix, valid_data, num_epoch, global_max_acc)

            epochs_list.append(epochs)
            train_losses_list.append(train_losses)
            valid_accs_list.append(valid_accs)
            valid_losses_list.append(valid_losses)
            max_accs_list.append(max_acc)
            if max_acc > global_max_acc:
                global_max_acc = max_acc
                global_max_acc_k = k

    plot_against_epochs(f"Training Loss against Epoch lr={lr}", "Training Loss",
        "k", k_list, epochs_list, train_losses_list)
    plot_against_epochs(f"Validation Accuracy against Epoch lr={lr}", "Validation Accuracy",
        "k", k_list, epochs_list, valid_accs_list)
    plot_against_epochs(f"Validation Loss against Epoch lr={lr}", "Validation Loss",
        "k", k_list, epochs_list, valid_losses_list)
    
    for i in range(len(k_list)):
        print(f"k={k_list[i]}: Max Validation Accuracy: {max_accs_list[i]}")

    print(f"Loading model with k={global_max_acc_k} and validation accuracy: {global_max_acc}")
    model = AutoEncoder(num_questions, global_max_acc_k)
    model.load_state_dict(torch.load("model.pth"))

    test_acc, test_loss = evaluate(model, zero_train_matrix, test_data)
    print(f"Test Accuracy: {test_acc}")

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
