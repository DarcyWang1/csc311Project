
from item_response import irt, sigmoid
from neural_network import AutoEncoder, train, load_data
from utils import *
from torch.autograd import Variable
import numpy as np
import torch
import random

def compute_acc(valid_data, pred):
    return np.sum((valid_data["is_correct"] == np.array(pred))) / len(valid_data["is_correct"])


def predict(zero_train_matrix, eval_data, theta, beta, model):
    # Start Evaluation
    pred_irt_raw = []
    pred_nn_raw = []

    pred_irt = []
    pred_nn = []

    for i in range(len(eval_data["user_id"])):
        # Input
        stud_id = eval_data["user_id"][i]
        ques_id = eval_data["question_id"][i]
        stud_answers = zero_train_matrix[stud_id]

        # Run IRT prediction
        x = (theta[stud_id] - beta[ques_id]).sum()
        p_a = sigmoid(x)
        pred_irt_raw.append(p_a)
        pred_irt.append(p_a >= 0.5)

        # Run NN prediction
        input = Variable(stud_answers).unsqueeze(0)
        output = model(input)
        guess = output[0][ques_id].item()
        pred_nn_raw.append(guess)
        pred_nn.append(guess >= 0.5)

    pred_ensemble = [((pred_irt_raw[i] + pred_nn_raw[i]) / 2) >= 0.5 for i in range(len(pred_irt))]
    return pred_irt, pred_nn, pred_ensemble


def run_evaluate(zero_train_matrix, eval_data, theta, beta, model):

    pred_irt, pred_nn, pred_ensemble = predict(zero_train_matrix, eval_data, theta, beta, model)    

    acc_ensemble = compute_acc(eval_data, pred_ensemble)
    acc_irt = compute_acc(eval_data, pred_irt)
    acc_nn = compute_acc(eval_data, pred_nn)

    return acc_irt, acc_nn, acc_ensemble


def main():

    # Set seed for reproducibility
    random.seed(0)
    torch.manual_seed(0)

    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    private_test_data = load_private_test_csv("../data")

    num_questions, zero_train_matrix, train_matrix, _, _ = load_data()
    num_features = zero_train_matrix.shape[1]

    # Train IRT
    lr = 0.01
    iter_num = 20
    theta, beta, _ = irt(sparse_matrix, val_data, lr, iter_num)

    # Train NN
    lr = 0.01
    k = 200
    num_epoch = 300
    model = AutoEncoder(num_features, num_questions, k)
    train(model, lr, 0, train_matrix, zero_train_matrix, val_data, num_questions, num_epoch, global_max_acc=0)
    model.load_state_dict(torch.load("model.pth"))

    # Start Evaluation
    acc_irt, acc_nn, acc_ensemble = run_evaluate(zero_train_matrix, val_data, theta, beta, model)

    print(f"IRT Validation Accuracy: {acc_irt}")
    print(f"NN Validation Accuracy: {acc_nn}")
    print(f"Ensemble Validation Accuracy: {acc_ensemble}")
    
    acc_irt, acc_nn, acc_ensemble = run_evaluate(zero_train_matrix, test_data, theta, beta, model)

    print(f"IRT Test Accuracy: {acc_irt}")
    print(f"NN Test Accuracy: {acc_nn}")
    print(f"Ensemble Test Accuracy: {acc_ensemble}")

    pred_irt, pred_nn, pred_ensemble = predict(zero_train_matrix, private_test_data, theta, beta, model)
    private_test_data_predict = private_test_data.copy()
    private_test_data_predict["is_correct"] = pred_irt.copy()
    
    save_private_test_csv(private_test_data_predict)


if __name__ == "__main__":
    main()
