import torch

# Training in transformed space
def train_loopGeneral(numeric_step, dataloader, model, loss_fn, optimizer, scheduler, sch):
    for batch, (X, y, tau) in enumerate(dataloader):
        preprocess, _ = model(X, tau) # Pass trough model

        XX = numeric_step(preprocess, tau) # Numerical step with kernel method
        
        preprocess_y, _ = model(y, tau) # preprocess y
        loss = loss_fn(XX, preprocess_y) # Loss is calculated in the new space

        error = loss.detach()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Scheduler
        if sch:
            scheduler.step()
    
    return error


def train_loopGeneralFrenkle(numeric_step, dataloader, model, loss_fn, optimizer, scheduler, sch, g):
    for batch, (X, y, tau) in enumerate(dataloader):
        preprocess, _ = model(X, tau) # Pass trough model

        XX = numeric_step(preprocess, tau, g) # Numerical step with kernel method
        
        preprocess_y, _ = model(y, tau) # preprocess y
        loss = loss_fn(XX, preprocess_y) # Loss is calculated in the new space

        error = loss.detach()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Scheduler
        if sch:
            scheduler.step()
    
    return error

# Testing in transformed space
def test_loopGeneral(numeric_step, dataloader, model, loss_fn):

    batches = len(dataloader)
    loss = 0
    with torch.no_grad():
        for batch, (X, y , tau) in enumerate(dataloader):
            preprocess, _ = model(X, tau) # Pass trough model
        
            # Numerical method step
            XX = numeric_step(preprocess, tau)
            

            preprocess_y, _ = model(y, tau) # Pass trough inverse model
            loss += loss_fn(XX, preprocess_y).item()

    loss /= batches

    return loss


def test_loopGeneralFrenkel(numeric_step, dataloader, model, loss_fn, g):

    batches = len(dataloader)
    loss = 0
    with torch.no_grad():
        for batch, (X, y , tau) in enumerate(dataloader):
            preprocess, _ = model(X, tau) # Pass trough model
        
            # Numerical method step
            XX = numeric_step(preprocess, tau, g)
            

            preprocess_y, _ = model(y, tau) # Pass trough inverse model
            loss += loss_fn(XX, preprocess_y).item()

    loss /= batches

    return loss