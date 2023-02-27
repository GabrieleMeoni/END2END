import numpy as np


def upsample_ds(ds, lb, N, M, tol, recursive=False):
       X_train_new, y_train_new = [], []
       total_events = lb.sum()
       for img, label in zip(ds, lb):
              if label == 0:
                     for _ in range(N):
                            X_train_new.append(img)
                            y_train_new.append(label)
              else:
                     for _ in range(M):
                            X_train_new.append(img)
                            y_train_new.append(label)

       X_train_new=np.asarray(X_train_new, dtype='uint8')
       y_train_new=np.asarray(y_train_new)


       if recursive:
              total_events = y_train_new.sum()
              print('Shape: ',X_train_new.shape[0] // 2)
              print('total_events: ',total_events)
              if not (abs(X_train_new.shape[0] // 2 - total_events) < tol):
                     X_train_new, y_train_new = upsample_ds(ds=ds, lb=lb, N=N, M=M+1, tol=100)

       return X_train_new, y_train_new   



if __name__ == '__main__':
       print('Executing main...')
       pass