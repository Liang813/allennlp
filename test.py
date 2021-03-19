from allennlp.data.fields import MultiLabelField
try:
  f = MultiLabelField([0, 0, 1], num_labels=3, skip_indexing=True)
  f.empty_field().as_tensor(None)
except Exception as e:
  print("TypeError")
  print(e)
