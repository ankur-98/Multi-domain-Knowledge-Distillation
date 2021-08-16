import torch
import os
import logging
from util.utils import accuracy, plotter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def log(filename):
    logging.basicConfig(level=logging.INFO)
    if filename: 
        handler = logging.handlers.WatchedFileHandler(os.path.join(".", "logs", filename+'.log'))
        formatter = logging.Formatter(logging.BASIC_FORMAT)
        handler.setFormatter(formatter)
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

def train(data, model, epochs, batch_size, optimizer, validate=True, print_epochs=True, plot=True, logfile=None):

    logger = logging.getLogger(__name__)
    log(logfile)
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

    return model, val_Acc.item()*100