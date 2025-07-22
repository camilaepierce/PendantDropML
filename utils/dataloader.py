import matplotlib.pyplot as plt
import numpy as np
import random
from torch import from_numpy, Tensor



class PendantDataLoader():

    def __init__(self, data, num_batches, random_seed=None, feat_fxn=lambda x: x["image"], lab_fxn=lambda x: x["surface_tension"], run_model=None):
        """
        Initializes a DataLoader Object with optional random seeding parameter
        """
        self.data = data
        seeded_random = random.Random(random_seed)
        self.order = seeded_random.sample(list(data.available_samples), len(data.available_samples))
        # print("Order: ", self.order)
        self.batches = np.array_split(self.order, num_batches)
        self.feat_fxn = feat_fxn
        self.lab_fxn = lab_fxn
        self.iter_idx = 0
        self.num_batches = num_batches
        # self.size = len(data)
        self.label_shape = (len(self.batches[0]),) + np.array(lab_fxn(data[self.order[0]])).shape

        if run_model:
            self.k_predictor = True
            self.pre_k_model = run_model
            self.feature_shape = (len(self.batches[0]),) + (80, 2) #HARDCODED
        else:
            self.feature_shape = (len(self.batches[0]),) + feat_fxn(data[self.order[0]]).shape
            self.k_predictor = False
    
    def __iter__(self):
        """
        Creates iterable (generating function) that returns (train_features, train_labels) 
        """
        for current_batch in self.batches:
            if self.k_predictor:
                self.pre_k_model.eval()
            features_batch = []
            labels_batch = []
            for sample_id in current_batch:
                sample = self.data[sample_id]
                if self.k_predictor:
                    features = self.feat_fxn(sample)
                    print(features.shape)
                    sigma_tensor = self.pre_k_model(Tensor.float(from_numpy(np.array(features))).unsqueeze(0)).detach().numpy()
                    features_batch.append(np.concatenate((features, sigma_tensor)))
                else:
                    features_batch.append(self.feat_fxn(sample))
                labels_batch.append(self.lab_fxn(sample))
            self.iter_idx += 1
            yield (Tensor.float(from_numpy(np.array(features_batch))), Tensor.float(from_numpy(np.array(labels_batch))))        
    

###################################
### Testing Torch DataLoader Use Case
###################################
if __name__ == "__main__":
    from extraction import PendantDropDataset
    import sys
    import torch
    sys.path.insert(0, '/home/camilapierce/Desktop/UNED/MLPendantDropUNED/')
    from models.elastic.Extreme2 import Extreme

    training_data = PendantDropDataset("data/elastic_mini/test_data_params", "data/elastic_mini/test_data_rz","data/elastic_mini/test_images",
                                       "data/elastic_mini/test_data_sigmas", clean_data=True, ignore_images=True)
    model = Extreme()
    model.load_state_dict(torch.load('model_weights/HuberCleanedMassive.pth', weights_only=True))
    train_dataloader = PendantDataLoader(training_data, feat_fxn=lambda x : x["coordinates"], lab_fxn=lambda x : x["Wo_Ar"]["Kmod"], num_batches=10, run_model=model)

    # print("dataloader length", len(train_dataloader))
    print("data from dataloader length", len(train_dataloader.data))
    print("data length", len(training_data))

    # for (X, y) in iter(train_dataloader):
    #     print(X[:1])
    #     print(y[:5])
    shuffled_order = train_dataloader.order

    # print(shuffled_order[0])
    # print(len(shuffled_order))
    # print(training_data[shuffled_order[0]])

    # Display image and label.
    dataLoader_iter = iter(train_dataloader)

    # print(dataLoader_iter)
    next_iter, iter2 = next(dataLoader_iter)
    # print(next_iter)
    print(len(next_iter))
    X, y = next(dataLoader_iter)
    print(len(X))
    print(len(y))
    print("ACUTAL", X.shape)
    print(y.shape)
    print("FEATURE SHAPE", train_dataloader.feature_shape)
    print("LABEL SHAPE", train_dataloader.label_shape)
    # print(train_features)
    # print(train_labels)
    # print(f"Feature batch shape: {train_features.shape}")
    # print(f"Labels batch shape: {train_labels.shape}")
    # img = train_features[0].squeeze()
    # label = train_labels[0]
    # plt.imshow(img, cmap="gray")
    # plt.show()
    # print(f"Label: {label}")