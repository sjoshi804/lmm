from datasets import load_dataset, load_from_disk, DatasetDict, Dataset

path = "/data/lmm/generated/v3_spatial_grid_multimodal"
dataset = load_from_disk(path)

# Copy old dataset, change image path
def copy(type):
    data = []
    for i in range(len(dataset[type])):
        temp = {'text' : dataset[type][i]['text'], 'prompt' : dataset[type][i]['prompt'], 
        'conversations' : dataset[type][i]['conversations'], 'image' : f'/home/allanz/data/grid/{type}/{i}.png'}
        data.append(temp)
    return data

train = copy('train')
validation = copy('validation')
test = copy('test')


def convert_to_dict_of_lists(data):
    result = {}
    for key in data[0].keys():  
        result[key] = [entry[key] for entry in data]
    return result

# Convert 'train', 'validation', and 'test' lists into dicts of lists
train_dict = convert_to_dict_of_lists(train)
validation_dict = convert_to_dict_of_lists(validation)
test_dict = convert_to_dict_of_lists(test)

# Create the datasets using from_dict
train_dataset = Dataset.from_dict(train_dict)
validation_dataset = Dataset.from_dict(validation_dict)
test_dataset = Dataset.from_dict(test_dict)

# Save the datasets
dataset_dict = DatasetDict({'train': train_dataset, 'validation': validation_dataset, 'test': test_dataset})
dataset_dict.save_to_disk('/home/allanz/data/datasets/v3.1_spatial_grid_multimodal')


