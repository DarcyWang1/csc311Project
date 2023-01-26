
from item_response import irt, sigmoid
from neural_network import AutoEncoder, train
from sklearn.impute import KNNImputer
from utils import *
from torch.autograd import Variable
import numpy as np
import torch


torch.manual_seed(0)


def compute_acc(valid_data, pred):
    return np.sum((valid_data["is_correct"] == np.array(pred))) / len(valid_data["is_correct"])


def run_evaluate(zero_train_matrix, eval_data, knn_mat, theta, beta, model, majority_vote=False):
    # Start Evaluation
    pred_knn_raw = []
    pred_irt_raw = []
    pred_nn_raw = []

    pred_knn = []
    pred_irt = []
    pred_nn = []

    for i in range(len(eval_data["user_id"])):

        # Input
        stud_id = eval_data["user_id"][i]
        ques_id = eval_data["question_id"][i]
        stud_answers = zero_train_matrix[stud_id]

        # Run KNN prediction
        knn_res = knn_mat[stud_id, ques_id]
        pred_knn_raw.append(knn_res)
        pred_knn.append(knn_res >= 0.5)

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

    if majority_vote:
        pred_ensemble = [True if [pred_knn[i], pred_irt[i], pred_nn[i]].count(True) > 1 else False for i in range(len(pred_irt))]
    else:
        pred_ensemble = [((pred_knn_raw[i] + pred_irt_raw[i] + pred_nn_raw[i]) / 3) >= 0.5 for i in range(len(pred_irt))]

    acc_ensemble = compute_acc(eval_data, pred_ensemble)
    acc_knn = compute_acc(eval_data, pred_knn)
    acc_irt = compute_acc(eval_data, pred_irt)
    acc_nn = compute_acc(eval_data, pred_nn)

    return acc_knn, acc_irt, acc_nn, acc_ensemble

def get_bag(sparse_matrix, bagsize: float):
    mask = (np.random.random(sparse_matrix.shape) > bagsize)
    bag = sparse_matrix * 1
    bag[mask] = np.nan
    return bag

def prepare_nn_data(sparse_matrix, bagsize:float):
    sparse_matrix = get_bag(sparse_matrix,bagsize)
    train_matrix = sparse_matrix.toarray()
    zero_train_matrix = train_matrix.copy()
    zero_train_matrix[np.isnan(train_matrix)] = 0
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix,train_matrix


def main():
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    _, num_questions = sparse_matrix.shape

    # Boostrap data for each model
    bag_size = 0.5
    knn_data = get_bag(sparse_matrix,bag_size)
    knn_train_matrix = knn_data.toarray()
    irt_matrix = get_bag(sparse_matrix,bag_size)
    nn_zero_train_matrix, nn_train_matrix = prepare_nn_data(sparse_matrix, bag_size)

    # Whether use average or take majority vote
    use_majority_vote = False

    # Train KNN
    k = 11
    nbrs = KNNImputer(n_neighbors=k)
    knn_mat = nbrs.fit_transform(knn_train_matrix)

    # Train IRT
    lr = 0.01
    iter_num = 20
    theta, beta, _, __ = irt(irt_matrix, val_data, lr, iter_num)

    # Train NN
    lr = 0.01
    k = 50
    num_epoch = 200
    model = AutoEncoder(num_questions, k)
    train(model, lr, 0, nn_train_matrix, nn_zero_train_matrix, val_data, num_epoch, global_max_acc=0)
    model = AutoEncoder(num_questions, k)
    model.load_state_dict(torch.load("model.pth"))

    # Start Evaluation
    for use_majority_vote in [True, False]:
        acc_knn, acc_irt, acc_nn, acc_ensemble = run_evaluate(nn_zero_train_matrix, val_data, knn_mat, theta, beta, model, use_majority_vote)

        if use_majority_vote:
            print(f"Using Majority Vote:")
        else:
            print(f"Using Average Vote:")

        print(f"KNN Validation Accuracy: {acc_knn}")
        print(f"IRT Validation Accuracy: {acc_irt}")
        print(f"NN Validation Accuracy: {acc_nn}")
        print(f"Ensemble Validation Accuracy: {acc_ensemble}")

        acc_knn, acc_irt, acc_nn, acc_ensemble = run_evaluate(nn_zero_train_matrix, test_data, knn_mat, theta, beta, model, use_majority_vote)

        print(f"KNN Test Accuracy: {acc_knn}")
        print(f"IRT Test Accuracy: {acc_irt}")
        print(f"NN Test Accuracy: {acc_nn}")
        print(f"Ensemble Test Accuracy: {acc_ensemble}")

def data2MMatrex(data):
    matrix = np.full(
        (max(data['user_id']) + 1, max(data['question_id']) + 1),
        np.nan)
    for i in range(len(data['user_id'])):
        matrix[data['user_id'][i], data['question_id'][i]] = \
        data['is_correct'][i]
    return matrix
def only_irt():
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    lr = 0.01
    iter_num = 20
    bag_size = 0.5
    num_bag = 3
    buffer = np.zeros(shape=sparse_matrix.shape)
    for i in range(num_bag):
        irt_matrix = get_bag(sparse_matrix, bag_size)
        theta, beta, _, __ = irt(irt_matrix, val_data, lr, iter_num)
        tb = (np.asarray([theta]).T @ (np.asarray([[1]] * beta.shape[0])).T) - \
             (np.asarray([[1]] * theta.shape[0]) @ np.asarray([beta]))

        buffer += sigmoid(tb)
    prediction = (buffer/num_bag)>0.5
    val_matrex = data2MMatrex(val_data)
    test_matrex = data2MMatrex(test_data)
    test_acc = np.sum(np.nan_to_num(1-np.abs(prediction-test_matrex),nan=0))/np.sum(np.nan_to_num(test_matrex>-1,nan=0))
    val_acc = np.sum(np.nan_to_num(1-np.abs(prediction-val_matrex),nan=0))/np.sum(np.nan_to_num(val_matrex>-1,nan=0))
    #print('test acc',test_acc)
    #print('val acc',val_acc)
    return test_acc,val_acc
    #print(buffer/num_bag>0.5)


if __name__ == "__main__":
    main()
    l=[]
    l2=[]
    for i in range(10):
        print(i)
        test_acc,val_acc = only_irt()
        l.append(test_acc)
        l2.append(val_acc)
    print(l,l2)
