# data reading and folds

df = pd.read_csv("documents/data/processed_data.csv")
df = df.drop(columns=["y2"], axis=1)
n_folds = len(pd.unique(df["cv_fold"]))

folds = {}

for fold in range(n_folds):
    fold_numbers = [i for i in range(n_folds)]
    fold_numbers.remove(fold)
    outer_validation = (fold + 1) % n_folds # Slightly different between implementations!!!
    print(f"fold: {fold}, outer_validation: {outer_validation}")
    holdout_df = df[df["cv_fold"] == fold]

    train_df = df[df["cv_fold"] != fold]
    outer_train_df_split = train_df[train_df["cv_fold"] != outer_validation]
    outer_validation_df = train_df[train_df["cv_fold"] == outer_validation]

    # add outer holdout, train, and validation data in a neural net data set

    inner_folds = []
    for i, inner_fold in enumerate(fold_numbers):
        if not inner_fold == fold:
            inner_test_df = train_df[train_df["cv_fold"] == inner_fold]
            inner_train_df = train_df[train_df["cv_fold"] != inner_fold]

            validation = fold_numbers[0] if i + 1 == n_folds - 1 else fold_numbers[i + 1]
            print(f"\tinner fold: {inner_fold}, inner validation: {validation}")
            inner_train_df_split = inner_train_df[inner_train_df["cv_fold"] != validation]
            inner_validation_df = inner_train_df[inner_train_df["cv_fold"] == validation]

            # add inner test, train, and validation data in a neural net data set

            inner_folds.append({"train": inner_train_data, "test": inner_test_data, "validation": inner_validation_data})


    folds[fold] = {"holdout": holdout_data, "train": outer_train_data, "validation": outer_validation_data,"inner_folds": inner_folds}

k = 0

# Inner cross val
# Adjust to receive neural net hyper params instead
def inner_cv(outer_fold: int, params: dict, num_boost_round: int, early_stopping_rounds: int,
             min_delta: float) -> float:
    scores = []
    for inner_fold in folds[outer_fold]["inner_folds"]:
        evals_result = {}

        # Train nn model
        model = trained_neural_net()

        # print(evals_result)
        # print(evals_result["test"]["error"][-1])

        scores.append(evals_result["test"]["rmse"][-1])

    return np.mean(scores)
