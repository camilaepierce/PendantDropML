from torch import nn



class FiveLayerCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=1, kernel_size=(1, 3, 3)),
            # torch.squeeze(),
            nn.ReLU(),
            nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(1, 9, 9)),
            nn.MaxPool3d(kernel_size=(1, 15, 15), stride=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(59924, 5000),
            nn.Linear(5000, 500),
            nn.Linear(500, 300),
            nn.Linear(300, 1)
        )
        self.name = "Five Layer CNN"

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).unsqueeze(2)
        # print("Forward x shape", x.shape)
        logits = self.linear_relu_stack(x)
        logits = logits.squeeze()
        return logits
    

# model = FiveLayerCNN().to(device)
# print(model)

# X = torch.rand(1, 28*28, device=device)
# logits = model(X)
# pred_probab = nn.Softmax(dim=1)(logits)
# y_pred = pred_probab.argmax(1)
# print(f"Predicted class: {y_pred}")

# print(f"Model structure: {model}\n\n")

# for name, param in model.named_parameters():
#     print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")





# learning_rate = 1e-3
# batch_size = 64
# epochs = 10
# testing_size = 20

# ##############################################################
# ### Custom Modules
# ##############################################################

# drop_dataset = PendantDropDataset("data/test_data_params", "data/test_data_rz","data/test_images")
# training_data, testing_data = drop_dataset.split_dataset(testing_size, 4)

# train_dataloader = PendantDataLoader(training_data, batch_size)
# test_dataloader = PendantDataLoader(testing_data, batch_size)

