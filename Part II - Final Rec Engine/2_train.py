# Import the required libraries
import pickle
import itertools
import numpy as np
from lightfm import LightFM, cross_validation
from lightfm.evaluation import auc_score


# Create a function to split to train and test
def split_data(df, ratio):
    train, test = cross_validation.random_train_test_split(df,
                                                           test_percentage=ratio,
                                                           random_state=5)
    
    return train, test


# Create a function to yield a grid of hyper-parameters
def sample_hyperparameters():
    while True:
        yield {
            "no_components": np.random.randint(10, 225),
            "learning_schedule": np.random.choice(["adagrad", "adadelta"]),
            "loss": np.random.choice(["bpr", "warp", "warp-kos"]),
            "learning_rate": np.random.exponential(0.05),
            "item_alpha": np.random.exponential(1e-8),
            "user_alpha": np.random.exponential(1e-8),
            "max_sampled": np.random.randint(5, 25),
            "num_epochs": np.random.randint(50, 150),
        }


# Create a function to optimize model's hyper-parameters
def random_search(train, test,
                  item_feats,
                  user_feats,
                  num_samples,
                  num_threads=1):
    for hyperparams in itertools.islice(sample_hyperparameters(), num_samples):
        num_epochs = hyperparams.pop("num_epochs")

        model = LightFM(**hyperparams)
        model.fit(train,
                  item_features=item_feats,
                  user_features=user_feats,
                  epochs=num_epochs,
                  num_threads=num_threads,
                  verbose=5)

        score = auc_score(model,
                          test,
                          train_interactions=train,
                          user_features=user_feats,
                          item_features=item_feats,
                          num_threads=num_threads).mean()

        hyperparams["num_epochs"] = num_epochs

        yield (score, hyperparams, model)


# Create a function to train the best model
def train_model(hyperparams, interactions, user_feats, item_feats):
    # Get the number of epochs
    epochs = hyperparams['num_epochs']
    hyperparams.pop('num_epochs')

    model = LightFM(**hyperparams)

    model = model.fit(interactions,
                      item_features=item_feats,
                      user_features=user_feats,
                      epochs=epochs,
                      num_threads=3,
                      verbose=5)

    return model


def main():
    print("\nLoading data...")

    # Load data from disk
    data = pickle.load(open("data.pkl", "rb"))
    users_csr = data['users_csr']
    investments_csr = data['investments_csr']
    ratings_csr = data['ratings_csr']

    # Split data into training and test sets
    train, test = split_data(ratings_csr, 0.1)

    print("\nOptimizing hybrid model's hyper-parameters...")

    # Optimize hyper-parameters
    (score, hyperparams, model) = max(random_search(train,
                                                    test,
                                                    investments_csr,
                                                    users_csr,
                                                    num_samples=50,
                                                    num_threads=4),
                                      key=lambda x: x[0])

    print("Best score {} at {}".format(score, hyperparams))

    print("\nTraining the best model on the full datasets...")

    # Train best model on the whole dataset
    model = train_model(hyperparams, ratings_csr, users_csr, investments_csr)

    # Save model to disk
    model_name = "hybrid_model.pkl"

    pickle.dump(model, open(model_name, 'wb'))

    print("\nBest trained model saved successfully to 'hybrid_model.pkl!")


if __name__ == '__main__':
    main()