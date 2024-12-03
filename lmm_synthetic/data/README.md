# Synthetic Data Generator

This script generates a synthetic text dataset based on parameters specified in a JSON configuration file.

## Usage

To generate a dataset, run the script with the path to your JSON configuration file and an optional output directory:

```sh
python data_generator.py --config path/to/config.json --output_dir path/to/output
```

##  Arguments
--config: Path to the JSON configuration file.
--output_dir: Path to save the generated dataset. Defaults to /home/sjoshi/lmm/data/generated/.

## Configuration File
The JSON configuration file should contain the following keys:

num_samples: A dictionary specifying the number of samples for train, validation, and test splits.
num_rows: The number of rows in the grid.
num_cols: The number of columns in the grid.
vocab: A list of vocabulary items to populate the grid.
vocab_subset_size: The size of the subset of the vocabulary to use.
num_questions: The number of questions to generate per grid.

## Sample Configuration

{
    "num_samples": {
        "train": 1000,
        "validation": 200,
        "test": 200
    },
    "num_rows": 5,
    "num_cols": 5,
    "vocab": ["apple", "banana", "cherry", "date", "elderberry"],
    "vocab_subset_size": 3,
    "num_questions": 5
}

# Sample Output

The generated dataset will contain text samples formatted as follows:

```
| apple | banana | cherry | apple | banana |
| date | elderberry | apple | cherry | date |
| banana | cherry | apple | elderberry | banana |
| date | apple | banana | cherry | elderberry |
| cherry | date | elderberry | apple | banana |
The grid above is size 5 by 5. Each cell contains an object from ['apple', 'banana', 'cherry'].
What object is in row 0, column 0? A: apple
What object is in row 1, column 2? A: apple
What object is in row 2, column 4? A: banana
What object is in row 3, column 1? A: apple
What object is in row 4, column 3? A: apple
```

# Output
The generated dataset will be saved in the specified output directory in a format compatible with Hugging Face's datasets library. The dataset will be split into train, validation, and test sets.

## Example
To generate a dataset using the sample configuration:

Save the sample configuration to config.json.
Run the script:

```bash
python data_generator.py --config config.json --output_dir ./generated_data
```

The dataset will be saved in the ./generated_data directory.

# Dependencies
Make sure to install the required dependencies:

```bash
pip install argparse json random datetime typing datasets pandas loguru tqdm
```

# Training Observations

- Able to train on 1M (population size 2M) for data generated using v3 config and get 90% accuracy on held out set
- Training on 1M (populatin size >> 2M) for 5x5 data of v2 config doesn't work as well: 26% accuracy on held out set
- Model trained on v3 can generalize to 5x5 grids of v2 quite well (63%)
- Perhaps to get good language model trained for this, we need some training curriculum:
    - this could be critical even in multimodal setting!
- However, no generalization (0%) to more complex vocab (imagenet vocab)
    - the model hallucinates and chooses examples that don't exist