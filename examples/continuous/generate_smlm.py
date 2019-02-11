from gsoftmax import datasets

kernel_sigmas = [64.0, 320.0, 640.0, 1920.0]

# Construct a generative model
# dataset = datasets.make_generative_model(n_beads=4, noise=10, max_iter=2,
#                                          random_state=None)
#
# for images, positions, weights in dataset:
#     print(images[0], positions[0], weights[0])
#
# import matplotlib.pyplot as plt
#
# fig, ax = plt.subplots(1, 1)
#
# ax.matshow(images[0])
# ax.scatter(positions[0, :, 0] / 100,
#            positions[0, :, 1] / 100,
#            s=weights[0] / 50,
#            color='red')
# ax.set_xlim([0, 64])
# ax.set_ylim([0, 64])
# plt.show()


dataset = datasets.SMLMDataset(name='MT0.N1.LD', modality='2D')
image, positions, weights = dataset[10000]

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1)

ax.matshow(dataset.X.mean(axis=0))
plt.show()

