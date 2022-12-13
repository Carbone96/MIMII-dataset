import pandas as pd


def separateLabels_fromTestset(test_set : pd.DataFrame) -> pd.DataFrame:
    labels = test_set['labels']
    labels = labels.astype(bool)
    # Retrieve the test_set datapoints
    return test_set.loc[ : , test_set.columns != 'labels' ], labels  # Fetch only the data (not the label!)

def shuffle_sets(train_set : pd.DataFrame, test_set : pd.DataFrame) -> pd.DataFrame:
    """ Shuffle the sets"""
    train_set = train_set.sample(frac=1,random_state=1)
    test_set = test_set.sample(frac=1,random_state=1)
    return train_set,test_set

def foo(train_set : pd.DataFrame, test_set : pd.DataFrame) -> pd.DataFrame:
    """ Shuffle the sets and return them + the labels"""
    train_set, test_set = shuffle_sets(train_set, test_set)
    test_set, labels = separateLabels_fromTestset(test_set)
    return train_set, test_set, labels


def main() -> None:
    train_set = pd.DataFrame([[1,2,3], [1,2,3], [1,2,3], [1,2,3] , [1,2,3] , [1,2,3]])
    test_set = pd.DataFrame([ [1,2,3,0] , [1,2,3,0] ], columns=['1','2','3','labels' ])

    train_set, test_set, labels = foo(train_set, test_set)

    print(labels)

if __name__  == '__main__':
    main()
