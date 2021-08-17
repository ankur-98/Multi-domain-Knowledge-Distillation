import torch
import os
import logging
from util.utils import accuracy, plotter, TemperatureAnnealing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def log(filename, logger, debug):
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
          validate=True, print_epochs=False, plot=True, logfile=None):

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

        if print_epochs:
            logger.debug(f"\nEpoch {epoch}")
            logger.debug(f'Train loss: {train_Loss.item()}')
            if validate: logger.debug(f'Validation loss: {val_Loss.item()}')
            logger.debug(f'Train acc: {train_Acc.item():.5%}')
            if validate: logger.debug(f'Validation acc: {val_Acc.item():.5%}')

    logger.info(f"\nEpoch {epoch+1}")
    logger.info(f'Train loss: {train_Loss.item()}')
    if validate: logger.info(f'Validation loss: {val_Loss.item()}')
    logger.info(f'Train acc: {train_Acc.item():.5%}')
    if validate: logger.info(f'Validation acc: {val_Acc.item():.5%}')

    if plot:
        plotter(train_ep_loss, val_ep_loss, "Loss")
        plotter(train_ep_acc, val_ep_acc, "Accuracy")

    logging.shutdown()
    return model, val_Acc.item() if validate else None

def student_train(Data, Student_Model, epochs, batch_size_ratio, optimizer, alpha=0.5, T=1.0,
                  validate=True, print_epochs=False, plot=True, logfile=None):

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

        if print_epochs:
            logger.debug(f"\nEpoch {epoch}")
            logger.debug(f'Train loss: {train_Loss.item()}')
            if validate: logger.debug(f'Validation loss: {val_Loss.item()}')
            logger.debug(f'Train acc: {train_Acc.item():.5%}')
            if validate: logger.debug(f'Validation acc: {val_Acc.item():.5%}')

    logger.info(f"\nEpoch {epoch+1}")
    logger.info(f'Train loss: {train_Loss.item()}')
    if validate: logger.info(f'Validation loss: {val_Loss.item()}')
    logger.info(f'Train acc: {train_Acc.item():.5%}')
    if validate: logger.info(f'Validation acc: {val_Acc.item():.5%}')

    if plot:
        plotter(train_ep_loss, val_ep_loss, "Loss")
        plotter(train_ep_acc, val_ep_acc, "Accuracy")

    logging.shutdown()
    return Student_Model, val_Acc.item() if validate else None

def student_train_KD_annealing(Data, Student_Model, epochs, batch_size_ratio, optimizer, alpha=0.5, T_max=1.0,
                  validate=True, print_epochs=False, plot=True, logfile=None):
    """
    Ref: Jafari, A., Rezagholizadeh, M., Sharma, P. and Ghodsi, A., 2021, April. Annealing Knowledge Distillation. 
         In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: 
         Main Volume (pp. 2493-2504).
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

        if print_epochs:
            logger.debug(f"\nEpoch {epoch}")
            logger.debug(f'Train loss: {train_Loss.item()}')
            if validate: logger.debug(f'Validation loss: {val_Loss.item()}')
            logger.debug(f'Train acc: {train_Acc.item():.5%}')
            if validate: logger.debug(f'Validation acc: {val_Acc.item():.5%}')

    logger.info(f"\nEpoch {epoch+1}")
    logger.info(f'Train loss: {train_Loss.item()}')
    if validate: logger.info(f'Validation loss: {val_Loss.item()}')
    logger.info(f'Train acc: {train_Acc.item():.5%}')
    if validate: logger.info(f'Validation acc: {val_Acc.item():.5%}')

    if plot:
        plotter(train_ep_loss, val_ep_loss, "Loss")
        plotter(train_ep_acc, val_ep_acc, "Accuracy")

    logging.shutdown()
    return Student_Model, val_Acc.item() if validate else None