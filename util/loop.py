import torch
import numpy as np
import os
import copy
import logging
from util.utils import accuracy, plotter, TemperatureAnnealing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def log(filename, logger, debug):
    """Logging the experiment results as log files."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level)
    if filename: 
        handler = logging.handlers.WatchedFileHandler(os.path.join(".", "logs", filename+'.log'))
        formatter = logging.Formatter(logging.BASIC_FORMAT)
        handler.setFormatter(formatter)
        logger.setLevel(level)
        logger.addHandler(handler)
    return logger

def train(data, model, epochs, batch_size, optimizer, 
          validate=True, print_epochs=False, plot=True, logfile=None, checkpoint_val_best=None):
    """Trains a model for classification task using negative log likelihood loss for a model returning log(softmax) predictions.

    Args: 
        data: The data dictionary to train and/or validate.
        model: The model to be used for training and/or validate.
        epochs: Total number of iterations.
        batch_size: The batch_size of training samples.
        optimizer: The optimizer object.
        validate: True for doing validation every epoch. False for not validating.
        print_epochs: True for printing the epochs.  False to run silently.
        plot: True to plot training and/or validation accuracy and loss curve.
        logfile: None by default resulting in no log file creation. Creates a log file with the filename passed.
        checkpoint_val_best: None by default returns final iteration values. Else,
                                "loss": Returns the model state and validation accuracy for lowest validation loss.
                                "accuracy": Returns the model state and validation accuracy for highest validation accuracy.

    Returns:

        model_best: The model state as per checkpoint_val_best value.
        display_valid_acc: The required validation accuracy value to be returned.
    """
    logger = logging.getLogger(__name__)
    logger = log(logfile, logger, print_epochs)
    model = model.to(device)
    train_ep_loss, val_ep_loss = [], []
    train_ep_acc, val_ep_acc = [], []
    
    for epoch in range(epochs):

        # TRAIN
        train_loss = []
        train_acc = []
        chunk_size = (len(data['x_train']) - 1) // batch_size + 1
        for i in range(chunk_size):
            x = torch.tensor(data['x_train'][i*batch_size: (i+1)*batch_size], dtype=torch.float32).to(device)
            y = torch.tensor(data['y_train'][i*batch_size: (i+1)*batch_size], dtype=torch.long).to(device)

            y_pred = model(x)
            loss = torch.nn.functional.nll_loss(y_pred, y, reduction='sum') / x.shape[0]
            acc = accuracy(y_pred, y)
            train_acc.append(acc.item())
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        train_Loss = torch.mean(torch.tensor(train_loss))
        train_Acc = torch.mean(torch.tensor(train_acc))
        train_ep_loss.append(train_Loss.item())
        train_ep_acc.append(train_Acc.item())

        # VALIDATE
        if validate:
            with torch.no_grad():
                val_loss = []
                val_acc = []
                val_batch_size = 300
                chunk_size = (len(data['x_valid']) - 1) // val_batch_size + 1
                for i in range(chunk_size):
                    x = torch.tensor(data['x_valid'][i*val_batch_size: (i+1)*val_batch_size], dtype=torch.float32).to(device)
                    y = torch.tensor(data['y_valid'][i*val_batch_size: (i+1)*val_batch_size], dtype=torch.long).to(device)

                    y_pred = model(x)
                    loss = torch.nn.functional.nll_loss(y_pred, y, reduction='sum') / x.shape[0]
                    acc = accuracy(y_pred, y)
                    val_acc.append(acc.item())
                    val_loss.append(loss.item())

            val_Loss = torch.mean(torch.tensor(val_loss))
            val_Acc = torch.mean(torch.tensor(val_acc))
            val_ep_loss.append(val_Loss.item())
            val_ep_acc.append(val_Acc.item())

            if checkpoint_val_best is None:
                model_best = model
            elif checkpoint_val_best == "loss" and val_Loss.item() <= np.amin(val_ep_loss):
                model_best = copy.deepcopy(model)
            elif checkpoint_val_best == "accuracy" and val_Loss.item() >= np.amax(val_ep_loss):
                model_best = copy.deepcopy(model)

        if print_epochs:
            logger.debug(f"\nEpoch {epoch}")
            logger.debug(f'Train loss: {train_Loss.item()}')
            if validate: logger.debug(f'Validation loss: {val_Loss.item()}')
            logger.debug(f'Train acc: {train_Acc.item():.5%}')
            if validate: logger.debug(f'Validation acc: {val_Acc.item():.5%}')

    if checkpoint_val_best is None or not validate:
        display_train_loss = train_Loss.item()
        display_valid_loss = val_Loss.item()
        display_train_acc = train_Acc.item()
        display_valid_acc = val_Acc.item()
    else:
        idx = np.argmin(val_ep_loss) if checkpoint_val_best == "loss" else np.argmax(val_ep_acc)
        display_train_loss = train_ep_loss[idx]
        display_valid_loss = val_ep_loss[idx]
        display_train_acc = train_ep_acc[idx]
        display_valid_acc = val_ep_acc[idx]
        
    if checkpoint_val_best is None or not validate: logger.info(f"\nEpoch {epoch+1}")
    else: logger.info(f"\nCheckpoint Epoch {idx+1} based on val_{checkpoint_val_best}")
    logger.info(f'Train loss: {display_train_loss}')
    if validate: logger.info(f'Validation loss: {display_valid_loss}')
    logger.info(f'Train acc: {display_train_acc:.5%}')
    if validate: logger.info(f'Validation acc: {display_valid_acc:.5%}')

    if plot:
        plotter(train_ep_loss, val_ep_loss, "Loss")
        plotter(train_ep_acc, val_ep_acc, "Accuracy")

    logging.shutdown()
    return model_best, display_valid_acc if validate else None


def student_train(Data, Student_Model, epochs, batch_size_ratio, optimizer, alpha=0.5, T=1.0,
                  validate=True, print_epochs=False, plot=True, logfile=None, checkpoint_val_best=None):
    """Trains a student model for classification task using cross entropy loss and knowledge distillation loss.

    Args: 
        Data: The data dictionary to train and/or validate. It should have the teacher infered logits too.
        Student_Model: The student model to be used for training and/or validate.
        epochs: Total number of iterations.
        batch_size_ratio: The ratio of the total data to be used as batch size for training.
        optimizer: The optimizer object.
        alpha: The alpha weighting of soft labels. Between [0,1]
        T: The temperature value for softening the logits for the soft label training.
        validate: True for doing validation every epoch. False for not validating.
        print_epochs: True for printing the epochs.  False to run silently.
        plot: True to plot training and/or validation accuracy and loss curve.
        logfile: None by default resulting in no log file creation. Creates a log file with the filename passed.
        checkpoint_val_best: None by default returns final iteration values. Else,
                                "loss": Returns the model state and validation accuracy for lowest validation loss.
                                "accuracy": Returns the model state and validation accuracy for highest validation accuracy.

    Returns:

        Student_Model_best: The model state as per checkpoint_val_best value.
        display_valid_acc: The required validation accuracy value to be returned.
    """

    logger = logging.getLogger(__name__)
    logger = log(logfile, logger, print_epochs)
    Student_Model = Student_Model.to(device)
    train_ep_loss, val_ep_loss = [], []
    train_ep_acc, val_ep_acc = [], []
    
    for epoch in range(epochs):

        # TRAIN
        train_loss = []
        train_acc = []

        loss, acc = torch.tensor(0.0).to(device), torch.tensor(0.0).to(device)
        for domain in Data.keys():
            data = Data[domain]
            batch_size = int(len(data['x_train']) * batch_size_ratio)    # batch_size_ratio = 1 means batch gradient descent
            chunk_size = (len(data['x_train']) - 1) // batch_size + 1
            for i in range(chunk_size):
                x = torch.tensor(data['x_train'][i*batch_size: (i+1)*batch_size], dtype=torch.float32).to(device)
                y = torch.tensor(data['y_train'][i*batch_size: (i+1)*batch_size], dtype=torch.long).to(device)
                z = data['z_train'][i*batch_size: (i+1)*batch_size].clone().detach().to(device)

                y_pred = Student_Model(x)
                z_pred = Student_Model.Z(x)

                KD_loss = torch.nn.KLDivLoss(reduction='sum')(torch.log_softmax(z_pred/T, dim=1), torch.softmax(z/T, dim=1)) * (T**2)
                CE_loss = torch.nn.functional.nll_loss(y_pred, y, reduction='sum')

                loss = ((1-alpha) * CE_loss + (alpha) * KD_loss) / x.shape[0]
                acc = accuracy(y_pred, y)
                train_loss.append(loss.item())
                train_acc.append(acc.item())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        train_Loss = torch.mean(torch.tensor(train_loss))
        train_Acc = torch.mean(torch.tensor(train_acc))
        train_ep_loss.append(train_Loss.item())
        train_ep_acc.append(train_Acc.item())

        # VALIDATE
        if validate:
            with torch.no_grad():
                val_loss = []
                val_acc = []

                loss, acc = torch.tensor(0.0).to(device), torch.tensor(0.0).to(device)
                for domain in Data.keys():
                    data = Data[domain]
                    val_batch_size = len(data['x_valid'])
                    chunk_size = (len(data['x_valid']) - 1) // val_batch_size + 1
                    for i in range(chunk_size):
                        x = torch.tensor(data['x_valid'][i*val_batch_size: (i+1)*val_batch_size], dtype=torch.float32).to(device)
                        y = torch.tensor(data['y_valid'][i*val_batch_size: (i+1)*val_batch_size], dtype=torch.long).to(device)
                        z = data['z_valid'][i*batch_size: (i+1)*batch_size].clone().detach().to(device)

                        y_pred = Student_Model(x)
                        z_pred = Student_Model.Z(x)
                
                        KD_loss = torch.nn.KLDivLoss(reduction='sum')(torch.log_softmax(z_pred/T, dim=1), torch.softmax(z/T, dim=1)) * (T**2)
                        CE_loss = torch.nn.functional.nll_loss(y_pred, y, reduction='sum')

                        loss = ((1-alpha) * CE_loss + (alpha) * KD_loss) / x.shape[0]
                        acc = accuracy(y_pred, y)
                        val_acc.append(acc.item())
                        val_loss.append(loss.item())

            val_Loss = torch.mean(torch.tensor(val_loss))
            val_Acc = torch.mean(torch.tensor(val_acc))
            val_ep_loss.append(val_Loss.item())
            val_ep_acc.append(val_Acc.item())

            if checkpoint_val_best is None:
                Student_Model_best = Student_Model
            elif checkpoint_val_best == "loss" and val_Loss.item() <= np.amin(val_ep_loss):
                Student_Model_best = copy.deepcopy(Student_Model)
            elif checkpoint_val_best == "accuracy" and val_Loss.item() >= np.amax(val_ep_loss):
                Student_Model_best = copy.deepcopy(Student_Model)

        if print_epochs:
            logger.debug(f"\nEpoch {epoch}")
            logger.debug(f'Train loss: {train_Loss.item()}')
            if validate: logger.debug(f'Validation loss: {val_Loss.item()}')
            logger.debug(f'Train acc: {train_Acc.item():.5%}')
            if validate: logger.debug(f'Validation acc: {val_Acc.item():.5%}')

    if checkpoint_val_best is None or not validate:
        display_train_loss = train_Loss.item()
        display_valid_loss = val_Loss.item()
        display_train_acc = train_Acc.item()
        display_valid_acc = val_Acc.item()
    else:
        idx = np.argmin(val_ep_loss) if checkpoint_val_best == "loss" else np.argmax(val_ep_acc)
        display_train_loss = train_ep_loss[idx]
        display_valid_loss = val_ep_loss[idx]
        display_train_acc = train_ep_acc[idx]
        display_valid_acc = val_ep_acc[idx]
        
    if checkpoint_val_best is None or not validate: logger.info(f"\nEpoch {epoch+1}")
    else: logger.info(f"\nCheckpoint Epoch {idx+1} based on val_{checkpoint_val_best}")
    logger.info(f'Train loss: {display_train_loss}')
    if validate: logger.info(f'Validation loss: {display_valid_loss}')
    logger.info(f'Train acc: {display_train_acc:.5%}')
    if validate: logger.info(f'Validation acc: {display_valid_acc:.5%}')

    if plot:
        plotter(train_ep_loss, val_ep_loss, "Loss")
        plotter(train_ep_acc, val_ep_acc, "Accuracy")

    logging.shutdown()
    return Student_Model_best, display_valid_acc if validate else None


def student_train_KD_annealing(Data, Student_Model, epochs, batch_size_ratio, optimizer, alpha=0.5, T_max=1.0,
                  validate=True, print_epochs=False, plot=True, logfile=None, checkpoint_val_best=None):
    """
    Trains a student model for classification task using knowledge distillation annealing technique.

    Ref: Jafari, A., Rezagholizadeh, M., Sharma, P. and Ghodsi, A., 2021, April. Annealing Knowledge Distillation. 
         In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: 
         Main Volume (pp. 2493-2504).

    Args: 
        Data: The data dictionary to train and/or validate. It should have the teacher infered logits too.
        Student_Model: The student model to be used for training and/or validate.
        epochs: Total number of iterations.
        batch_size_ratio: The ratio of the total data to be used as batch size for training.
        optimizer: The optimizer object.
        alpha: The alpha weighting of soft labels. Between [0,1]
        T_max: The max temperature value for softening the logits for the soft label training.
        validate: True for doing validation every epoch. False for not validating.
        print_epochs: True for printing the epochs.  False to run silently.
        plot: True to plot training and/or validation accuracy and loss curve.
        logfile: None by default resulting in no log file creation. Creates a log file with the filename passed.
        checkpoint_val_best: None by default returns final iteration values. Else,
                                "loss": Returns the model state and validation accuracy for lowest validation loss.
                                "accuracy": Returns the model state and validation accuracy for highest validation accuracy.

    Returns:

        Student_Model_best: The model state as per checkpoint_val_best value.
        display_valid_acc: The required validation accuracy value to be returned.
    """
    logger = logging.getLogger(__name__)
    logger = log(logfile, logger, print_epochs)
    Student_Model = Student_Model.to(device)
    train_ep_loss, val_ep_loss = [], []
    train_ep_acc, val_ep_acc = [], []
    
    for epoch in range(epochs):

        # TRAIN
        train_loss = []
        train_acc = []

        loss, acc = torch.tensor(0.0).to(device), torch.tensor(0.0).to(device)
        phi = TemperatureAnnealing(T_max, epochs)
        for domain in Data.keys():
            data = Data[domain]
            batch_size = int(len(data['x_train']) * batch_size_ratio)    # batch_size_ratio = 1 means batch gradient descent
            chunk_size = (len(data['x_train']) - 1) // batch_size + 1
            for i in range(chunk_size):
                x = torch.tensor(data['x_train'][i*batch_size: (i+1)*batch_size], dtype=torch.float32).to(device)
                y = torch.tensor(data['y_train'][i*batch_size: (i+1)*batch_size], dtype=torch.long).to(device)
                z = data['z_train'][i*batch_size: (i+1)*batch_size].clone().detach().to(device)

                y_pred = Student_Model(x)
                z_pred = Student_Model.Z(x)

                T = phi(epoch)
                KD_anneal_loss = torch.nn.functional.mse_loss(z_pred, z*T, reduction='sum') / x.shape[0]
                CE_loss = torch.nn.functional.nll_loss(y_pred, y, reduction='sum') / x.shape[0]

                loss = KD_anneal_loss if T > 1 else (1-alpha) * CE_loss + alpha * KD_anneal_loss
                acc = accuracy(y_pred, y)
                train_loss.append(loss.item())
                train_acc.append(acc.item())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        train_Loss = torch.mean(torch.tensor(train_loss))
        train_Acc = torch.mean(torch.tensor(train_acc))
        train_ep_loss.append(train_Loss.item())
        train_ep_acc.append(train_Acc.item())

        # VALIDATE
        if validate:
            with torch.no_grad():
                val_loss = []
                val_acc = []

                loss, acc = torch.tensor(0.0).to(device), torch.tensor(0.0).to(device)
                for domain in Data.keys():
                    data = Data[domain]
                    val_batch_size = len(data['x_valid'])
                    chunk_size = (len(data['x_valid']) - 1) // val_batch_size + 1
                    for i in range(chunk_size):
                        x = torch.tensor(data['x_valid'][i*val_batch_size: (i+1)*val_batch_size], dtype=torch.float32).to(device)
                        y = torch.tensor(data['y_valid'][i*val_batch_size: (i+1)*val_batch_size], dtype=torch.long).to(device)
                        z = data['z_valid'][i*batch_size: (i+1)*batch_size].clone().detach().to(device)

                        y_pred = Student_Model(x)
                        z_pred = Student_Model.Z(x)

                        T = phi(epoch)
                        KD_anneal_loss = torch.nn.functional.mse_loss(z_pred, z*T, reduction='sum') / x.shape[0]
                        CE_loss = torch.nn.functional.nll_loss(y_pred, y, reduction='sum') / x.shape[0]

                        loss = KD_anneal_loss if T > 1 else (1-alpha) * CE_loss + alpha * KD_anneal_loss
                        acc = accuracy(y_pred, y)
                        val_acc.append(acc.item())
                        val_loss.append(loss.item())

            val_Loss = torch.mean(torch.tensor(val_loss))
            val_Acc = torch.mean(torch.tensor(val_acc))
            val_ep_loss.append(val_Loss.item())
            val_ep_acc.append(val_Acc.item())

            if checkpoint_val_best is None:
                Student_Model_best = Student_Model
            elif checkpoint_val_best == "loss" and val_Loss.item() <= np.amin(val_ep_loss):
                Student_Model_best = copy.deepcopy(Student_Model)
            elif checkpoint_val_best == "accuracy" and val_Loss.item() >= np.amax(val_ep_loss):
                Student_Model_best = copy.deepcopy(Student_Model)

        if print_epochs:
            logger.debug(f"\nEpoch {epoch}")
            logger.debug(f'Train loss: {train_Loss.item()}')
            if validate: logger.debug(f'Validation loss: {val_Loss.item()}')
            logger.debug(f'Train acc: {train_Acc.item():.5%}')
            if validate: logger.debug(f'Validation acc: {val_Acc.item():.5%}')

    if checkpoint_val_best is None or not validate:
        display_train_loss = train_Loss.item()
        display_valid_loss = val_Loss.item()
        display_train_acc = train_Acc.item()
        display_valid_acc = val_Acc.item()
    else:
        idx = np.argmin(val_ep_loss) if checkpoint_val_best == "loss" else np.argmax(val_ep_acc)
        display_train_loss = train_ep_loss[idx]
        display_valid_loss = val_ep_loss[idx]
        display_train_acc = train_ep_acc[idx]
        display_valid_acc = val_ep_acc[idx]
        
    if checkpoint_val_best is None or not validate: logger.info(f"\nEpoch {epoch+1}")
    else: logger.info(f"\nCheckpoint Epoch {idx+1} based on val_{checkpoint_val_best}")
    logger.info(f'Train loss: {display_train_loss}')
    if validate: logger.info(f'Validation loss: {display_valid_loss}')
    logger.info(f'Train acc: {display_train_acc:.5%}')
    if validate: logger.info(f'Validation acc: {display_valid_acc:.5%}')

    if plot:
        plotter(train_ep_loss, val_ep_loss, "Loss")
        plotter(train_ep_acc, val_ep_acc, "Accuracy")

    logging.shutdown()
    return Student_Model_best, display_valid_acc if validate else None