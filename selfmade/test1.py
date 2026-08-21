from nasbench import api

rec = './nasbench_full.tfrecord'
inp = 'input'
out = 'output'
conv1x1 = 'conv1x1-bn-relu'
conv3x3 = 'conv3x3-bn-relu'
maxpool3x3 = 'maxpool3x3'

nasbench = api.NASBench(rec)

matrix=[[0, 1, 1, 1, 0, 1, 0],    # input layer
        [0, 0, 0, 0, 0, 0, 1],    # 1x1 conv
        [0, 0, 0, 0, 0, 0, 1],    # 3x3 conv
        [0, 0, 0, 0, 1, 0, 0],    # 5x5 conv (replaced by two 3x3's)
        [0, 0, 0, 0, 0, 0, 1],    # 5x5 conv (replaced by two 3x3's)
        [0, 0, 0, 0, 0, 0, 1],    # 3x3 max-pool
        [0, 0, 0, 0, 0, 0, 0]]    # output layer

matrix = [[0, 1],
          [0, 0]]

# matrix =[[0, 1, 1, 1, 1, 1, 1],
#          [0, 0, 1, 1, 1, 1, 1],
#          [0, 0, 0, 1, 1, 1, 1],
#          [0, 0, 0, 0, 1, 1, 1],
#          [0, 0, 0, 0, 0, 1, 1],
#          [0, 0, 0, 0, 0, 0, 1],
#          [0, 0, 0, 0, 0, 0, 0]]

# ops=[inp, conv1x1, conv3x3, conv3x3, conv3x3, maxpool3x3, out]
ops=[inp, out]

model_spec = api.ModelSpec(matrix=matrix, ops=ops)
print(model_spec.matrix)
print(model_spec.ops)

data = nasbench.query(model_spec)
print(data)
print(nasbench.get_budget_counters())

print('\nGetting all metrics for the same Inception-like model.')
fixed_metrics, computed_metrics = nasbench.get_metrics_from_spec(model_spec)
print(fixed_metrics)
for epochs in nasbench.valid_epochs:
    for repeat_index in range(len(computed_metrics[epochs])):
        data_point = computed_metrics[epochs][repeat_index]
        print('Epochs trained %d, repeat number: %d' % (epochs, repeat_index + 1))
        print(data_point)
# print('\nIterating over unique models in the dataset.')
# for unique_hash in nasbench.hash_iterator():
#     fixed_metrics, computed_metrics = nasbench.get_metrics_from_hash(
#         unique_hash)
#     print(fixed_metrics)
    
# print(len(nasbench.hash_iterator()))