import matplotlib.pyplot as plt
import numpy as np
import random
from torch import from_numpy, Tensor



class PendantDataLoader():

    def __init__(self, data, num_batches, random_seed=None, feat_fxn=lambda x: x["image"], lab_fxn=lambda x: x["surface_tension"]):
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
        self.feature_shape = (num_batches,) + feat_fxn(data[self.order[0]]).shape
        self.label_shape = (num_batches,) + lab_fxn(data[self.order[0]]).shape
    
    def __iter__(self):
        """
        Creates iterable (generating function) that returns (train_features, train_labels) 
        """
        for current_batch in self.batches:
            features_batch = []
            labels_batch = []
            for sample_id in current_batch:
                sample = self.data[sample_id]
                try:
                    features_batch.append(self.feat_fxn(sample))
                except:
                    print(sample)
                labels_batch.append(self.lab_fxn(sample))
            self.iter_idx += 1
            yield (Tensor.float(from_numpy(np.array(features_batch))), Tensor.float(from_numpy(np.array(labels_batch))))        
    

###################################
### Testing Torch DataLoader Use Case
###################################
if __name__ == "__main__":
    from extraction import PendantDropDataset

    training_data = PendantDropDataset("data/test_data_params", "data/test_data_rz","data/test_images")

    train_dataloader = PendantDataLoader(training_data, num_batches=10, shuffle=True)

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
    next_iter, iter2 = next(dataLoader_iter)
    print(len(next_iter))
    print()
    print(train_dataloader.feature_shape)
    # print(train_features)
    # print(train_labels)
    # print(f"Feature batch shape: {train_features.shape}")
    # print(f"Labels batch shape: {train_labels.shape}")
    # img = train_features[0].squeeze()
    # label = train_labels[0]
    # plt.imshow(img, cmap="gray")
    # plt.show()
    # print(f"Label: {label}")