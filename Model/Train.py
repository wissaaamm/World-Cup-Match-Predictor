import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from Network import Network


df = pd.read_csv("Datasets/dataset.csv")
df["date"] = pd.to_datetime(df["date"])

FEATURES = ["loc_home", "loc_neutral", "loc_away", "elo_diff", "form_diff", "rest_diff"]
TARGETS  = ["home_goals", "away_goals"]

train = df[df["date"] <  "2018-01-01"]
test  = df[df["date"] >= "2018-01-01"]

X_train = torch.tensor(train[FEATURES].values, dtype=torch.float32)
y_train = torch.tensor(train[TARGETS].values,  dtype=torch.float32)
X_test  = torch.tensor(test[FEATURES].values,  dtype=torch.float32)
y_test  = torch.tensor(test[TARGETS].values,   dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)

# ── training ──────────────────────────────────────────────────────────────────

epochs = 100
lr = 1e-3

model     = Network()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.PoissonNLLLoss(log_input=False)

for epoch in range(epochs):
    model.train()
    train_loss = 0.0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            preds     = model(X_test)
            test_loss = criterion(preds, y_test)
            mae       = torch.mean(torch.abs(preds - y_test))

        print(f"Epoch {epoch+1:3d} | "
              f"train loss: {train_loss/len(train_loader):.4f} | "
              f"test loss: {test_loss:.4f} | "
              f"MAE: {mae:.4f}")

# ── save ──────────────────────────────────────────────────────────────────────

torch.save(model.state_dict(), "Models/goal_predictor.pt")
print("Model saved.")