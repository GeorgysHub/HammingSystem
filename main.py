import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

class MemoryRecallSystem:
    def __init__(self, training_data):
        self.num_patterns = len(training_data)
        self.size = len(training_data[0])
        self.weight_matrix = np.zeros((self.size, self.size))

        # Обучение сети
        for pattern in training_data:
            pattern = np.array(pattern)
            self.weight_matrix += np.outer(pattern, pattern)
        np.fill_diagonal(self.weight_matrix, 0)

    def reconstruct(self, input_pattern, max_steps=100):
        input_pattern = np.array(input_pattern)
        for _ in range(max_steps):
            new_pattern = np.sign(np.dot(self.weight_matrix, input_pattern))
            new_pattern[new_pattern == 0] = 1
            if np.array_equal(new_pattern, input_pattern):
                break
            input_pattern = new_pattern
        return input_pattern


def add_noise_to_pattern(pattern, noise_ratio=0.2):
    noisy_pattern = np.copy(pattern)
    num_noisy_bits = int(noise_ratio * len(pattern))
    noise_indices = np.random.choice(len(pattern), num_noisy_bits, replace=False)
    noisy_pattern[noise_indices] = -noisy_pattern[noise_indices]
    return noisy_pattern


def show_comparison(original, noisy, recovered):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    def display_image(axis, img, title):
        axis.imshow(img, cmap='gray')
        axis.set_title(title)
        axis.axis('off')

    display_image(axes[0], original, "Original")
    display_image(axes[1], noisy, "Noisy")
    display_image(axes[2], recovered, "Recovered")

    plt.show()


def img_to_vector(image_file, dimensions=(128, 128)):
    img = Image.open(image_file).convert('L')
    img = img.resize(dimensions)
    img_array = np.array(img)
    binary_array = (img_array > 128).astype(int)
    binary_array[binary_array == 0] = -1
    return binary_array.flatten()


def vector_to_img(vector, dimensions=(128, 128)):
    img_data = ((vector + 1) // 2 * 255).reshape(dimensions).astype(np.uint8)
    return Image.fromarray(img_data)


if __name__ == "__main__":
    folder_path = r"Z:/PyProjects/systemHamming/images"
    files = [
        os.path.join(folder_path, "image1.png"),
        os.path.join(folder_path, "image2.png"),
        os.path.join(folder_path, "image3.png"),
        os.path.join(folder_path, "image4.png")
    ]

    img_dimensions = (128, 128)

    training_patterns = [img_to_vector(file, dimensions=img_dimensions) for file in files]

    recall_network = MemoryRecallSystem(training_patterns)

    test_patterns = training_patterns[:]

    noise_ratio = 0.3
    noisy_patterns = [add_noise_to_pattern(pattern, noise_ratio=noise_ratio) for pattern in test_patterns]

    print("Classification using Memory Recall System:")
    for i, noisy_pattern in enumerate(noisy_patterns):
        restored_pattern = recall_network.reconstruct(noisy_pattern)
        print(f"Noisy input vector matches pattern {i + 1}")
        show_comparison(
            vector_to_img(test_patterns[i], dimensions=img_dimensions),
            vector_to_img(noisy_pattern, dimensions=img_dimensions),
            vector_to_img(restored_pattern, dimensions=img_dimensions)
        )
