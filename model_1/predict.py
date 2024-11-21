import argparse
import pandas as pd
import torch
import sys
from sklearn import metrics
from sklearn.metrics import accuracy_score
from torch_geometric.loader import DataLoader
from codes.create_data import TestbedDataset
from codes.train_pipeline import predicting
from models import gat

cellfile = "cell_line_GAT"
testfile = "6271_drug_synergy_GAT"
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-model",
        "--model",
        type=str,
        default="gat",
    )
    parser.add_argument(
        "-name", "--name", type=str, default="GAT", help="Model Name"
    )
    parser.add_argument(
        "-data",
        "--data",
        type=str,
        default="6271_drug_synergy_GAT",
        help="Input data for prediction",
    )
    args = parser.parse_args()
    run(model_name=args.name, model_type=args.model, testfile=args.data)




def run(testfile, model_name, model_type):
    # creat data
    drug1_data_test = TestbedDataset(root="data", dataset=testfile + "_drug1")
    drug2_data_test = TestbedDataset(root="data", dataset=testfile + "_drug2")

    drug1_loader_test = DataLoader(drug1_data_test, batch_size = 64, shuffle = None)
    drug2_loader_test = DataLoader(drug2_data_test, batch_size = 64, shuffle = None)

    df_test = pd.read_csv("data/" + testfile + ".csv")
    df_smile = pd.read_csv("data/smiles.csv")
    drug_to_smile = {row["name"]: row["smile"] for i, row in df_smile.iterrows()}
    smile_to_drug = {v: k for k, v in drug_to_smile.items()}

    # CPU or GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # load model
    print("\nLoading {} model: {}".format(model_type.upper(), model_name))
    if model_type == "gat":
        model = gat.GATNet().to(device)
    

    path = "trained_model/" + model_name
    try:
        model.load_state_dict(torch.load(path))
    except:
        print("Wrong model type!")
        sys.exit()

    y_true, prob, y_pred = predicting(
        model, device, drug1_loader_test, drug2_loader_test
    )

    print("\nModel predictions: ")
    for i, row in df_test.iterrows():
        print(
            "{} drug1: {}, drug2: {}, cell: {}, True label: {} | Prediction: {:.0f} (score={:.3f})".format(
                i + 1,
                smile_to_drug[row.drug1],
                smile_to_drug[row.drug2],
                row.cell,
                row.label,
                y_pred[i],
                prob[i],
            )
        )


        # save results
        file_results = 'result/predict' +  '.txt'
        with open(file_results, 'a') as f:
            f.write(f'i: {i}, drug1: {smile_to_drug[row.drug1]}, drug2: {smile_to_drug[row.drug2]},cell: {row.cell}, True label: {row.label}, Prediction: {y_pred[i]}, score: {prob[i]}\n')
        


    print("y_true:", y_true)
    print("y_pred:", y_pred)
    print("prob:", prob)
    ACC = accuracy_score(y_true, y_pred)
    print("Accuracy:", ACC)   




    df_pred = df_test.copy()
    df_pred["prediction"] = y_pred
    df_pred["probability"] = prob
    j_pred = df_pred.to_json(orient="records")

    n_ones_true = len(df_pred[df_pred.label == 1])
    n_ones_pred = len(df_pred[df_pred.prediction == 1])
    ncorrect = df_pred[df_pred.prediction == df_pred.label].prediction.count()
    print("\nNumber of 1s: True={}, Predicted={}".format(n_ones_true, n_ones_pred))
    print(
        "Number of 0s: True={}, Predicted={}".format(
            len(df_pred) - n_ones_true, len(df_pred) - n_ones_pred
        )
    )
    print(
        "\Correct predictions: {}/{} = {:.2%}".format(
            ncorrect, len(df_pred), ncorrect / len(df_pred)
        )
    )

    # write predictions to disk
    with open("data/processed/predictions", "w") as f:
        f.write(j_pred)
    print("\nPredictions written to data/processed/predictions.json \n")



if __name__ == "__main__":
    main()



