# CustomDatasetClasses

This repository provides custom `Dataset` classes implemented in PyTorch to facilitate loading and preprocessing of various datasets for machine learning tasks.

## Features

- **Custom Dataset Implementations**: Tailored `Dataset` classes for specific data formats and structures, enabling efficient data handling in PyTorch.
- **Data Preprocessing**: Includes common preprocessing steps such as normalization, augmentation, and transformation to prepare data for training and evaluation.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/imaditya123/CustomDatasetClasses.git
2. **Navigate to the project directory**:
   ```bash
   cd CustomDatasetClasses
3. **Install the required dependencies**:
   Ensure that PyTorch is installed in your environment. Install other dependencies using:
   
   ```bash
   pip install -r requirements.txt

<!-- ## Usage

1. **Import the custom dataset classes**:
    ```bash
    python3 -m spacy download en_core_web_sm
    from custom_datasets import CustomDataset1, CustomDataset2
2. **Initialize the dataset**:
    ```bash
    train_dataset = CustomDataset1(data_dir='path/to/data', transform=transformations)
3. **Create a DataLoader**:
    ```bash
    from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
Iterate over the DataLoader in your training loop:
for data, labels in train_loader:
    # Training code here -->

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository.**
2. **Create a new branch**:
    ```bash
    git checkout -b feature/your-feature-name
3. **Commit your changes**:
   ```bash
   git commit -m 'Add some feature'
4. **Push to the branch**:
    ```bash
    git push origin feature/your-feature-name
5. **Open a Pull Request.**


## License

This project is licensed under the Apache-2.0 License. See the [LICENSE](https://github.com/imaditya123/CustomDatasetClasses/blob/main/LICENSE) file for details.
