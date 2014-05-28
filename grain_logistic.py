import numpy as np
import sklearn.linear_model
import sys
import grain_common

if __name__ == "__main__":
  
  train_set_id = int(sys.argv[1])
  test_set_id = int(sys.argv[2])

  train_name = grain_common.filename_data(train_set_id, 'train')
  X, Y = grain_common.load_raw_data(train_name)
  X, Y = grain_common.preprocess_X(X, Y, 1) # train_set_id)

  lr = sklearn.linear_model.LogisticRegression()
  lr.fit(X,Y)

  
  test_name = grain_common.filename_data(test_set_id, 'train')
  X2, Y2 = grain_common.load_raw_data(test_name)
  X2, Y2 = grain_common.preprocess_X(X2, Y2, 1) #train_set_id)
  print "MAP: %f" % ( lr.score(X,Y))
