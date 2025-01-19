import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

#import requests

#url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
#response = requests.get(url)

# Save the content to a local file
#with open('pima-indians-diabetes.csv', 'wb') as file:
#    file.write(response.content)
#
#print("Data downloaded and saved as 'pima-indians-diabetes.csv'")

dataset = np.loadtxt('/Users/till/Documents/School/Robotics/LunarLander/pima-indians-diabetes.csv', delimiter=",")
X = dataset[:,0:8]
y = dataset[:,8]

#convert to torch tensors from numpay arrays
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1,1)

model = nn.Sequential(
    nn.Linear(8,12),
    nn.ReLU(),
    nn.Linear(12,8),
    nn.ReLU(),
    nn.Linear(8,1),
    nn.Sigmoid()
)

loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

n_epochs = 100
batch_size = 10

for epoch in range(n_epochs):
    for i in range(0, len(X), batch_size):
        Xbatch = X[i:i+batch_size]
        y_pred = model(Xbatch)
        ybatch = y[i:i+batch_size]
        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Finished epoch {epoch}, latest loss {loss}")


# compute accuracy (no_grad is optional)
with torch.no_grad():
    y_pred = model(X)
 
accuracy = (y_pred.round() == y).float().mean()
print(f"Accuracy {accuracy}")

# make class predictions with the model
predictions = (model(X) > 0.5).int()
for i in range(5):
    print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))