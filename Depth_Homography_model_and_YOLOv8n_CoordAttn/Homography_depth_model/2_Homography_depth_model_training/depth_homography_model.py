"""
if pip is broken:
python -m ensurepip --upgrade
python -m pip install --upgrade pip

pip install scikit-learn

"""
#!/usr/bin/env python3
import glob, os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# ----------------------------
# CNP model (fixed encoder)
# ----------------------------
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1+9, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128)
        )
    def forward(self, x, y):
        # x: (N,1), y: (N,9)
        inp = torch.cat([x, y], dim=1)          # -> (N,10)
        return self.net(inp).mean(dim=0, keepdim=True)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1+128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 9)
        )
    def forward(self, x, r):
        inp = torch.cat([x, r.repeat(x.size(0),1)], dim=1)
        return self.net(inp)

class CNP(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    def forward(self, x_context, y_context, x_target):
        r = self.encoder(x_context, y_context)
        return self.decoder(x_target, r)

# ----------------------------
# Data + utils
# ----------------------------
def load_data():
    files = sorted(glob.glob("homography_depth_*.npz"))
    depths, Hs = [], []
    for f in files:
        d = float(np.load(f)['depth'])
        H = np.load(f)['homography'].flatten()
        depths.append(d); Hs.append(H)
    return np.array(depths).reshape(-1,1), np.array(Hs)

def normalize(X, Y):
    xm, xr = X.mean(), X.max()-X.min()
    Xn = 2*(X-xm)/xr
    ym, ys = Y.mean(axis=0), Y.std(axis=0)
    Yn = (Y-ym)/ys
    return Xn, Yn, (xm,xr), (ym,ys)

def denorm(Yn, params):
    ym, ys = params
    return Yn*ys + ym

def predict(model, depth, norm_params):
    (xm,xr), y_params = norm_params
    x_norm = torch.tensor([[2*(depth-xm)/xr]], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        pred = model(x_norm, torch.zeros_like(x_norm).repeat(1,9), x_norm)
    return denorm(pred.numpy().squeeze(), y_params).reshape(3,3)

# ----------------------------
# Main
# ----------------------------
def main():
    depths, Hs = load_data()
    Xn, Yn, xp, yp = normalize(depths, Hs)

    model = CNP()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    X = torch.tensor(Xn, dtype=torch.float32)
    Y = torch.tensor(Yn, dtype=torch.float32)

    loss_history = []

    # Replace your existing training loop with this:
    for epoch in range(800):
        pred = model(X, Y, X)
        loss = nn.MSELoss()(pred, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        if epoch % 250 == 0:
            print(f"Epoch {epoch} — Loss {loss.item():.4f}")

    print("\nPredictions on train set:")
    for d in depths.flatten():
        print(f"{d}m →\n{predict(model, d, (xp,yp))}\n")

    H = predict(model, 8.0, (xp,yp))
    print("Predicted H @8m:\n", H)

    ir = cv2.imread("ir_test.png")
    rgb= cv2.imread("vs_test.png")
    warp = cv2.warpPerspective(ir, H.astype(np.float32), (rgb.shape[1], rgb.shape[0]))
    alpha = 0.7
    fused = cv2.addWeighted(rgb, alpha, warp, 0.5, 0)
    cv2.imwrite("fused_test.png", fused)
    print("Saved fused_test.png")
    #cv2.imshow('Fused Image', fused)
    #cv2.waitKey(0)

    # Plot training loss
    plt.figure(figsize=(8, 6))
    plt.plot(loss_history, linewidth=2)

    plt.title('CNP Training Loss', fontname='Times New Roman', fontsize=20)
    plt.xlabel('Epoch', fontname='Times New Roman', fontsize=16)
    plt.ylabel('MSE Loss', fontname='Times New Roman', fontsize=16)

    plt.xticks(fontname='Times New Roman', fontsize=12)
    plt.yticks(fontname='Times New Roman', fontsize=12)

    plt.grid(alpha=0.3, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('training_loss.png', dpi=300)
    plt.close()

    print("Saved training_loss.png")

if __name__=="__main__":
    main()
