from model.randomforest import RandomForest
from model.SGD import SGD


def model_predict(data, df, name):
    """Call train/predict/print_results on each model via the uniform BaseModel interface."""
    print("RandomForest")
    model = RandomForest("RandomForest", data.get_embeddings(), data.get_type())
    model.train(data)
    model.predict(data.X_test)
    model.print_results(data)

    print("SGD")
    model = SGD("SGD", data.get_embeddings(), data.get_type())
    model.train(data)
    model.predict(data.X_test)
    model.print_results(data)


def model_evaluate(model, data):
    model.print_results(data)
