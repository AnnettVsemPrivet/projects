# основано на функциях из гитхаба
# https://github.com/SheezaShabbir/Time-series-Analysis-using-LSTM-RNN-and-GRU

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

device = "cuda" if torch.cuda.is_available() else "cpu"


# Создаем необходимые переменные
def Create_Dataloaders(
        cluster: int,
        data_stand: np.ndarray,
        y_pred_cl: np.ndarray,
        batchs_size: int,
        bootstrap_times: int,
        predict_period: int,
        test_set: float,
        val_set: float,
        seed: int = 0,
) -> None:

    global batch_size
    global train_loader
    global test_loader_one
    global val_loader

    test = test_set + val_set
    batch_size = batchs_size

    # отделяем датасет с кластером
    nn_df = data_stand[y_pred_cl == cluster].reset_index(drop=True)
    # отделяем тест перед бутстрэпом, т.к. в тесте не должно быть повторов
    nn_df, df_test = train_test_split(np.array(nn_df), test_size=test,
                                      random_state=seed)
    nn_df = pd.DataFrame(nn_df).reset_index(drop=True)
    df_test = pd.DataFrame(df_test).reset_index(drop=True)

    # bootstrap part
    np.random.seed(0)
    # набираем индексы с возвращением чтобы увеличить выборку
    lst = np.random.choice(range(len(nn_df)), len(nn_df) * bootstrap_times,
                           replace=True)
    # увеличиваем выборку
    nn_df = nn_df.iloc[lst, :].reset_index(drop=True)

    # splitting the data into test, validation, and train sets
    X_train, y_train = np.array(nn_df.iloc[:, :-predict_period]), \
        np.array(nn_df.iloc[:, -predict_period:])
    X, y = np.array(df_test.iloc[:, :-predict_period]), \
        np.array(df_test.iloc[:, -predict_period:])

    # validation - for training, test - for final quality evaluation
    X_val, X_test, y_val, y_test = train_test_split(X, y,
                                                    test_size=test_set / test,
                                                    random_state=seed)

    # loading the data into DataLoaders
    train_features = torch.Tensor(X_train)
    train_targets = torch.Tensor(y_train)
    val_features = torch.Tensor(X_val)
    val_targets = torch.Tensor(y_val)
    test_features = torch.Tensor(X_test)
    test_targets = torch.Tensor(y_test)

    train = TensorDataset(train_features, train_targets)
    val = TensorDataset(val_features, val_targets)
    test = TensorDataset(test_features, test_targets)

    train_loader = DataLoader(train, batch_size=batch_size,
                              shuffle=False, drop_last=True)
    val_loader = DataLoader(val, batch_size=batch_size,
                            shuffle=False, drop_last=True)
    test_loader_one = DataLoader(test, batch_size=1,
                                 shuffle=False, drop_last=True)

    # plot samples for manual check
    x_lim = X_train.shape[1] + y_train.shape[1]

    plt.figure(figsize=(10, 4))
    for i in range(X_train.shape[0]):
        label = '_nolegend_' if i! = 1 else 'train sample'
        plt.plot(pd.concat([pd.DataFrame(X_train[i]),
                            pd.DataFrame(y_train[i])],
                           axis=0).reset_index(drop=True),
                 "-", alpha=0.2, c='b', label=label)
    plt.xlim(0, x_lim)
    plt.ylim(-4, 4)
    plt.axvline(x=X_train.shape[1], color='r',
                linestyle="--", label='prediction horizon')
    plt.title('training set')
    plt.legend(loc='upper left')
    plt.show()

    plt.figure(figsize=(10, 4))
    for i in range(X_test.shape[0]):
        label = '_nolegend_' if i != 1 else 'test sample'
        plt.plot(pd.concat([pd.DataFrame(X_test[i]),
                            pd.DataFrame(y_test[i])],
                           axis=0).reset_index(drop=True),
                 "-", alpha=0.2, c='c', label=label)
    for i in range(X_val.shape[0]):
        label = '_nolegend_' if i != 1 else 'validation sample'
        plt.plot(pd.concat([pd.DataFrame(X_val[i]),
                            pd.DataFrame(y_val[i])],
                           axis=0).reset_index(drop=True),
                 "-", alpha=0.2, c='m', label=label)
    plt.xlim(0, x_lim)
    plt.ylim(-4, 4)
    plt.axvline(x=X_test.shape[1], color='r',
                linestyle="--", label='prediction horizon')
    plt.title('testing set')
    plt.legend(loc='upper left')
    plt.show()


# LSTM model class
class LSTMModel(nn.Module):
    """LSTMModel class extends nn.Module class and works
        as a constructor for LSTMs.

       LSTMModel class initiates a LSTM module based on PyTorch's
       nn.Module class. It has only two methods, namely init()
       and forward(). While the init() method initiates the model
       with the given input parameters, the forward() method defines
       how the forward propagation needs to be calculated.
       Since PyTorch automatically defines back propagation,
       there is no need to define back propagation method.

       Attributes:
           hidden_dim (int): The number of nodes in each layer
           layer_dim (str): The number of layers in the network
           lstm (nn.LSTM): The LSTM model constructed with
                the input parameters.
           fc (nn.Linear): The fully connected layer to convert the final state
                           of LSTMs to our desired output shape.

    """
    def __init__(self, input_dim, hidden_dim,
                 layer_dim, output_dim, dropout_prob):
        """The __init__ method that initiates a LSTM instance.

        Args:
            input_dim (int): The number of nodes in the input layer
            hidden_dim (int): The number of nodes in each layer
            layer_dim (int): The number of layers in the network
            output_dim (int): The number of nodes in the output layer
            dropout_prob (float): The probability of nodes being dropped out

        """
        super(LSTMModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, layer_dim,
            batch_first=True, dropout=dropout_prob
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """The forward method takes input tensor x and does forward propagation

        Args:
            x (torch.Tensor): The input tensor of the shape
                (batch size, sequence length, input_dim)

        Returns:
            torch.Tensor: The output tensor of the shape
                (batch size, output_dim)

        """
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0),
                         self.hidden_dim, device=x.device).requires_grad_()

        # Initializing cell state for first input with zeros
        c0 = torch.zeros(self.layer_dim, x.size(0),
                         self.hidden_dim, device=x.device).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through
        # time (BPTT). If we dont, we'll backprop all the way to the start
        # even after going through another batch forward propagation by
        # passing in the input, hidden state, and cell state into the model.
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Reshaping the outputs in the shape of
        # (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape
        # (batch_size, output_dim)
        out = self.fc(out)

        return out


# Training class
class Optimization:
    """Optimization is a helper class that allows training,
        validation, prediction.

    Optimization is a helper class that takes model, loss function,
    optimizer function learning scheduler (optional), early stopping (optional)
    as inputs. In return, it provides a framework to train and validate
    the models, and to predict future values based on the models.

    Attributes:
        model (LSTMModel): Model class created for the type of RNN
        loss_fn (torch.nn.modules.Loss): Loss function to calculate the losses
        optimizer (torch.optim.Optimizer): Optimizer function to optimize
            the loss function
        train_losses (list[float]): The loss values from the training
        val_losses (list[float]): The loss values from the validation
        last_epoch (int): The number of epochs that the models is trained
    """
    def __init__(self, model, loss_fn, optimizer):
        """
        Args:
            model (LSTMModel): Model class created for the type of RNN
            loss_fn (torch.nn.modules.Loss): Loss function to calculate
                the losses
            optimizer (torch.optim.Optimizer): Optimizer function to optimize
                the loss function
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []

    def train_step(self, x, y):
        """The method train_step completes one step of training.

        Given the features (x) and the target values (y) tensors, the method
        completes one step of the training. First, it activates the train mode
        to enable back prop. After generating predicted values (yhat) by doing
        forward propagation, it calculates the losses by using
        the loss function. Then, it computes the gradients by doing
        back propagation and updates the weights by calling step() function.

        Args:
            x (torch.Tensor): Tensor for features to train one step
            y (torch.Tensor): Tensor for target values to calculate losses

        """
        # Sets model to train mode
        self.model.train()

        # Makes predictions
        yhat = self.model(x)

        # Computes loss
        loss = self.loss_fn(y, yhat)

        # Computes gradients
        loss.backward()

        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss
        return loss.item()

    def train(self, train_loader, val_loader,
              batch_size=50, n_epochs=50, n_features=30):
        """The method train performs the model training

        The method takes DataLoaders for training and validation datasets,
        batch size for mini-batch training, number of epochs to train,
        and number of features as inputs. Then, it carries out the training
        by iteratively calling the method train_step for n_epochs times.
        If early stopping is enabled, then it checks the stopping condition
        to decide whether the training needs to halt before n_epochs steps.
        Finally, it saves the model in a designated file path.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader that
                stores training data
            val_loader (torch.utils.data.DataLoader): DataLoader that
                stores validation data
            batch_size (int): Batch size for mini-batch training
            n_epochs (int): Number of epochs, i.e., train steps, to train
            n_features (int): Number of feature columns

        """
        # model_path = f'{self.model}_' +
        # f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.view([batch_size, -1, n_features]).to(device)
                y_batch = y_batch.to(device)
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:

                    x_val = x_val.view([batch_size, -1, n_features]).to(device)

                    y_val = y_val.to(device)
                    self.model.eval()

                    yhat = self.model(x_val)
                    val_loss = self.loss_fn(y_val, yhat).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

            if (epoch <= 10) | (epoch % 10 == 0):
                print(
                    f"[{epoch} / {n_epochs}] Training loss: "
                    + f"{training_loss:.4f}\tValidation loss: "
                    + f"{validation_loss:.4f}"
                )

    def evaluate(self, test_loader, batch_size=1, n_features=1):
        """The method evaluate performs the model evaluation

        The method takes DataLoaders for the test dataset, batch size for
        mini-batch testing, and number of features as inputs. Similar to
        the model validation, it iteratively predicts the target values
        and calculates losses. Then, it returns two lists that hold
        the predictions and the actual values.

        Note:
            This method assumes that the prediction from the previous step
            is available at the time of the prediction, and only does
            one-step prediction into the future.

        Args:
            test_loader (torch.utils.data.DataLoader): DataLoader
                that stores test data
            batch_size (int): Batch size for mini-batch training
            n_features (int): Number of feature columns

        Returns:
            list[float]: The values predicted by the model
            list[float]: The actual values in the test set.

        """
        with torch.no_grad():
            predictions = []
            values = []
            for x_test, y_test in test_loader:
                x_test = x_test.view([batch_size, -1, n_features]).to(device)
                y_test = y_test.to(device)
                self.model.eval()
                yhat = self.model(x_test)
                yhat = yhat.cpu().data.numpy()
                predictions.append(yhat)
                y_test = y_test.cpu().data.numpy()
                values.append(y_test)

        return predictions, values

    def plot_losses(self):
        """The method plots the calculated loss values for
            training and validation
        """
        plt.style.use('ggplot')
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
        plt.close()


# main settings
def General_Settings(
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        layer_dim: int,
        dropout: float,
        n_epochs: int,
        learning_rate: float,
        weight_decay: float,
):

    model_name = 'lstm'
    model_params = {'input_dim': input_dim,
                    'hidden_dim' : hidden_dim,
                    'layer_dim' : layer_dim,
                    'output_dim' : output_dim,
                    'dropout_prob' : dropout}
    model = get_model(model_name, model_params)
    loss_fn = nn.MSELoss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                           weight_decay=weight_decay)

    opt = Optimization(model=model.to(device), loss_fn=loss_fn,
                       optimizer=optimizer)
    opt.train(train_loader, val_loader, batch_size=batch_size,
              n_epochs=n_epochs, n_features=input_dim)
    opt.plot_losses()
    model_dict = opt.model.state_dict()

    # за счет использования batch_size=1 -
    # получается использовать в тесте все элементы.
    predictions, values = opt.evaluate(
        test_loader_one,
        batch_size=1,
        n_features=input_dim
    )
    return predictions, values, model_dict


def get_model(model, model_params):
    models = {
        "lstm": LSTMModel,
    }
    return models.get(model.lower())(**model_params)


# format the predictions
def inverse_transform(scaler, df, columns):
    for col in columns:
        df[col] = scaler.inverse_transform(df[col])
    return df


def format_predictions(predictions, values, df_test, scaler):
    vals = np.concatenate(values, axis=0).ravel()
    preds = np.concatenate(predictions, axis=0).ravel()
    df_result = pd.DataFrame(data={"value": vals, "prediction": preds})
    # df_result = inverse_transform(
    # scaler, df_result, [["value", "prediction"]])
    return df_result


# error metrics
def Quality_Check(predictions, y_test):
    print('avg R2: %.2f, avg MSE: %.2f' %
          tuple(np.mean(pd.DataFrame([[r2_score(y_true[0], y_predict[0]),
                                       mean_squared_error(y_true[0],
                                                          y_predict[0])]
                                      for (y_true, y_predict)
                                      in zip(y_test, predictions)]),
                        axis=0)))

    print('med R2: %.2f, med MSE: %.2f' %
          tuple(np.median(pd.DataFrame([[r2_score(y_true[0], y_predict[0]),
                                         mean_squared_error(y_true[0],
                                                            y_predict[0])]
                                        for (y_true, y_predict)
                                        in zip(y_test, predictions)]),
                          axis=0)))
