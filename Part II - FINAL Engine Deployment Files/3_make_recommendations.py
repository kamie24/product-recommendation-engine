# Import the required libraries
import pickle
import numpy as np
import pandas as pd
from lightfm import LightFM, cross_validation
from lightfm.evaluation import auc_score


# Create a function to make recommendations for individual users
def sample_recommendation_user(model,
                               interactions,
                               user_feats,
                               item_feats,
                               uid,
                               user_dict, 
                               item_dict,
                               threshold=0,
                               nrec_items=5,
                               show=True):
    
    _, n_items = interactions.shape
    user_x = user_dict[uid]
    scores = pd.Series(model.predict(user_x,
                                     np.arange(n_items),
                                     item_features=item_feats,
                                     user_features=user_feats
                                    )
                      )
    scores.index = interactions.columns
    scores = list(pd.Series(scores.sort_values(ascending=False).index))
    
    known_items = list(pd.Series(interactions.loc[uid,:] \
                                 [interactions.loc[uid,:] > threshold].index).sort_values(ascending=False))
    
    scores = [x for x in scores if x not in known_items]
    return_score_list = scores[0:nrec_items]
    known_items = list(pd.Series(known_items,
                                 dtype='object').apply(lambda x: 'id: ' + str(x) + ' - ' + item_dict[x]))
    scores = list(pd.Series(return_score_list).apply(lambda x: 'id: ' + str(x) + ' - ' + item_dict[x]))
    if show == True:
        print("User: " + str(uid))
        print("Known Likes:")
        counter = 1
        for i in known_items:
            print(str(counter) + '- ' + i)
            counter+=1

        print("\n Recommended Items:")
        counter = 1
        for i in scores:
            print(str(counter) + '- ' + i)
            counter+=1
    return scores


def main():
    print("\nLoading model...")

    # Load data from disk
    data = pickle.load(open("data.pkl", "rb"))
    users_csr = data['users_csr']
    investments_csr = data['investments_csr']
    user_dict = data['users_dictionary']
    item_dict = data['items_dictionary']
    rating_piv = data['rating_pivoted']

    # Load model from disk
    model = pickle.load(open("hybrid_model.pkl", "rb"))

    print("\nMaking recommendations...")

    # Create recommendations for each user
    recommendations = {}
    for user, n in user_dict.items():
        scores = sample_recommendation_user(model,
                                            rating_piv,
                                            users_csr,
                                            investments_csr,
                                            user,
                                            user_dict,
                                            item_dict,
                                            show=False)
        recommendations[user] = scores


    # Create a df from the dictionary
    recommendations = pd.DataFrame.from_dict(recommendations).T

    # Save to CSV
    recommendations.to_csv('recommendations.csv', index_label='User_ID')

    print("\nRecommendations saved successfully to 'recommendations.csv'!")


if __name__ == '__main__':
    main()