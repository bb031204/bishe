import pickle

# 替换'your_file.pkl'为你的文件路径
file_path = '/home/fjt/Bigscity-LibCity-master/raw_data/humidity/test.pkl'

with open(file_path, 'rb') as file:
    data = pickle.load(file)


print('x',data['x'].shape[2])
print('y',data['y'].shape)
print('context',data['context'].shape)


def load_data(file_path):
    # 打开.pkl文件并加载内容
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data
train_file = '/home/fjt/Bigscity-LibCity-master/raw_data/humidity/trn.pkl'
test_file = '/home/fjt/Bigscity-LibCity-master/raw_data/humidity/test.pkl'
val_file = '/home/fjt/Bigscity-LibCity-master/raw_data/humidity/val.pkl'


train_data = load_data(train_file)
test_data = load_data(test_file)
val_data = load_data(val_file)
x_train = train_data.get('x')
y_train = train_data.get('y')
x_test = test_data.get('x')
y_test = test_data.get('y')
x_val = val_data.get('x')
y_val = val_data.get('y')
print("Training Data Loaded:", x_train is not None and y_train is not None)
print("Test Data Loaded:", x_test is not None and y_test is not None)
print("Validation Data Loaded:", x_val is not None and y_val is not None)