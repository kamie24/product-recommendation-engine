# Import the required libraries
import pandas as pd
import pickle
from scipy.sparse import csr_matrix


# Create a function to load data and specific features
def load_data(fname, features=None):
    return pd.read_csv(fname, usecols=features)


# Create an item dictionary for future references
def create_item_dict(df, iuid, name):
    # Init a dict
    item_dict = {}

    # Create a new df with the iuid and the name/title
    temp = df[[iuid, name]].sort_values(iuid).reset_index()

    # Populate the dict
    for i in range(temp.shape[0]):
        item_dict[(temp.loc[i, iuid])] = temp.loc[i, name]
        
    # Remove name
    df.drop(columns=[name], inplace=True)

    return item_dict, df


# Create a function to preprocess user and item data
def prepare_data(df, uid):
    # One-hot encode features
    df = pd.get_dummies(df, columns=df.columns[1:])

    # Sort by id and reset index
    df = df.sort_values(uid).reset_index().drop('index', axis=1)

    # Convert to csr matrix
    csr = csr_matrix(df.drop(uid, axis=1).values)

    # print(df.shape)

    return csr


# Create a function to find missing records between two dfs
def find_missing(df1, df2, uid):
    missing = list(set(df1[uid].unique()).symmetric_difference(set(df2[uid].unique())))
    
    return missing


# Create a function to preprocess interactions and create a user id dictionary
def prepare_interactions(interactions, users, uid, items, iuid):
    # Pivot table to re-organize data
    pivoted = pd.pivot_table(interactions,
                             index=uid,
                             columns=iuid,
                             values='inferred_user_rating')

    # Find which users are not present in both datasets
    missing = find_missing(interactions, users, uid)

    # Create a user dictionary
    all_ids = list(pivoted.index)
    user_dict = {}
    counter = 0 
    for i in all_ids:
        user_dict[i] = counter
        counter += 1

    # Add the missing users into the pivoted interactions dataset
    pivoted = pivoted.append(pd.DataFrame(index=missing))

    # Find which items are not present in the interactions dataset
    missing = find_missing(interactions, items, iuid)

    # Create a list of all items
    all_items = (pivoted.columns.tolist() + missing)

    # Add the missing items into the pivoted interactions
    pivoted = pd.concat([pivoted, pd.DataFrame(columns=all_items)]).fillna(0)

    # print(pivoted.shape)

    # Convert to csr matrix
    csr = csr_matrix(pivoted.values)

    return user_dict, pivoted, csr


def main():
    print("\nLoading and preparing datasets...")

    # Load Data
    users = load_data('user-attributes-1.csv')
    investments = load_data('deal-attributes-1.csv')
    ratings = load_data('user-interactions-1.csv',
                        features=['user_id', 'investment_id',
                                  'inferred_user_rating'])

    # Create an item dictionary
    item_dict, investments = create_item_dict(investments,
                                              'investment_id',
                                              'title')

    # Prepare data
    users_csr = prepare_data(users, 'user_id')
    investments_csr = prepare_data(investments, 'investment_id')

    # Prepare interactions data and create a user dictionary
    user_dict, rating_piv, ratings_csr = prepare_interactions(ratings,
                                                              users,
                                                              'user_id',
                                                              investments,
                                                              'investment_id')


    # Create a dictionary of data
    data = dict()
    data['users_csr'] = users_csr
    data['investments_csr'] = investments_csr
    data['ratings_csr'] = ratings_csr
    data['users_dictionary'] = user_dict
    data['items_dictionary'] = item_dict
    data['rating_pivoted'] = rating_piv

    # Save data to disk
    pickle.dump(data, open("data.pkl","wb"))

    print("\nData prepared and saved successfully to 'data.pkl'!")


if __name__ == '__main__':
    main()