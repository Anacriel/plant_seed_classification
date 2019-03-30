import pandas as pd
from sklearn import svm, preprocessing
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def main():
    features_df = pd.read_csv("features.csv", sep='\t')
    features_df.head()

    y = features_df['label'].values
    print(y)
    y = preprocessing.LabelEncoder().fit_transform(y)
    print(y)

    frame = features_df.drop(['label'], axis=1)

    X = frame.values
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, random_state=7, test_size=0.3)

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #X_train = preprocessing.normalize(X_train, norm='l2')
    #X_test = preprocessing.normalize(X_test, norm='l2')

    ######################

    #clf = svm.SVC(kernel='rbf', C=10.0,
    #                                 gamma=0.9, decision_function_shape='ovr')
    #clf.fit(X_train, Y_train)
    #x_predict = clf.predict(X_test)
    #print(x_predict)

    #clf = OneVsRestClassifier(svm.LinearSVC())
    #clf.fit(X_train, Y_train)
    #print(clf.score(X_test, Y_test))

    clf = svm.SVC(kernel='rbf', C = 10.0,
                                      gamma=0.9, decision_function_shape='ovr')
    clf.fit(X_train, Y_train)
    #print(clf.score(X_test, Y_test))
    x_predict = clf.predict(X_test)
    print(clf.score(X_test, Y_test))
    print(x_predict)

    #clf = OneVsRestClassifier(svm.SVC(kernel='rbf', C = 10.0,
    #                                  gamma=0.9, decision_function_shape='ovr'))
    #clf.fit(X_train, Y_train)
    #print(clf.score(X_test, Y_test))

    accuracy = accuracy_score(Y_test, x_predict)
    print("accuracy: {0:.2f}%".format(accuracy*100))

    test_df = pd.read_csv("test_data.csv", sep='\t')
    titles = test_df['title']
    test_frame = test_df.drop(['title'], axis=1)
    X = test_frame.values
    X = scaler.transform(X)
    x_result = clf.predict(X)
    print(x_result)

    dic = dict(zip(titles, x_result))

    for key, value in dic.items():
        print("{} - {}".format(key, value))


if __name__ == '__main__':
    main()
