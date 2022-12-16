from Seq_model import SequentialModel
from sklearn import datasets
from keras.utils import np_utils
from dense_layer import DenseLayer
from sklearn import preprocessing
from sklearn.utils import shuffle

if __name__ == '__main__':


    iris = datasets.load_iris()
    X = iris.data

    # normalize and restrcture the data for easier processing
    X = preprocessing.normalize(X)
    X = X.reshape(X.shape[0], 1, 4)

    # turn the Y values to categorical structure: [0 0 1] instead of 2
    Y = np_utils.to_categorical(iris.target)

    # shuffles the data to get all labels in the different datasets, train,validation and test
    X, Y = shuffle(X, Y, random_state=0)

    model = SequentialModel()

    # create layers, using sigmoid for output
    layer1 = DenseLayer(64, "relu")
    layer2 = DenseLayer(16, 'relu')
    layer3 = DenseLayer(16, 'relu')
    layer4 = DenseLayer(8, 'relu')
    output = DenseLayer(3, 'sigmoid')

    model.add_layer(layer1)
    model.add_layer(layer2)
    model.add_layer(layer3)
    model.add_layer(layer4)
    model.add_layer(output)

    # saving some of the dataset as testset, the data is shuffled otherwise this is a bad way to split the data
    model.train(X[:130], Y[:130], 1000)
    model.save('NN_model.yaml')
    model.load('NN_model.yaml')
    model.print_model()
    predictions = model.predict(X[131:149],Y[131:149])
    print(predictions)





