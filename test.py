from allennlp.data.fields import MultiLabelField
import traceback
try:
  f = MultiLabelField([0, 0, 1], num_labels=3, skip_indexing=True)
  f.empty_field().as_tensor(None)
except Exception as e:
  traceback.print_exc(file=open('/script/allennlp3721-buggy.txt','w+'))
