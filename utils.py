from tensorflow.keras import utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


class Gallery:
    def __init__(self, model, x, y, size, num_classes):
        self.model = model
        self.num_classes = num_classes
        self.x = x
        if len(y.shape) > 1 and y.shape[1]!=1:
            y = y.argmax(axis=1)
        self.y = y
        self.size = size

    def show(self, n=20, classList=None, showClass='all', columns=6, figsize=(16, 16), order='random', tightLayout=True):
      if showClass!='all' and classList == None:
          raise ValueError('`classList` must be provided to display a particular class.')
      if showClass !='all':
        if showClass in classList:
          showClassIndex = classList.index(showClass)
        else:
          raise ValueError('No `{}` found in `"class list"`.'.format(showClass))
        if order == 'random':
          imageIndex = random.sample(list(np.where(self.y==showClassIndex)[0]),n)
        elif order == 'first':
          imageIndex = list(np.where(self.y==showClassIndex)[0])[:n]
        elif order == 'last':
            imageIndex = list(np.where(self.y==showClassIndex)[0])[-n:]
        else:
          raise ValueError('`Order` must be one of `"first"`, `"last"` or `"random"`.')
      else:
        if order == 'random':
            imageIndex = [random.randint(0, self.size) for _ in range(n)]
        elif order == 'first':
            imageIndex = list(range(n))
        elif order == 'last':
            imageIndex = list(range(self.size - n, self.size))
        else:
            raise ValueError('`Order` must be one of `"first"`, `"last"` or `"random"`.')

      rows = (n + columns - 1) / columns
      fig = plt.figure(figsize=figsize)

      for i in range(n):
          img = self.x[imageIndex[i]]
          ax = fig.add_subplot(rows, columns, i + 1, xticks=[], yticks=[])
          if classList:
              ax.set_title('True:%s' % (classList[int(self.y[imageIndex[i]])]))
          if np.max(img) < 20:  # threshold of 20
              min_val = np.min(img)
              max_val = np.max(img)
              img_unnormalized = (img - min_val) / (max_val - min_val)
              plt.imshow(img_unnormalized)
          else:
              plt.imshow(img)

      if tightLayout:
          fig.tight_layout()
      plt.axis('off')

    def showMisclassified(self, x_test=None, y_test=None, n=20, classList=None, showClass='all', columns=6, figsize=(16, 16),
                          order='random', tightLayout=True):
      if showClass!='all' and classList == None:
        raise ValueError('`classList` must be provided to display a particular class.')
      if x_test is None:
          x_test = self.x
      if y_test is None:
          y_test = self.y
      if len(y_test.shape) < 2 or y_test.shape[1] != self.num_classes:
          y_test = utils.to_categorical(y_test, self.num_classes)
      y_pred = self.model.predict(x_test)
      Y_pred = y_pred.argmax(axis=1)
      y_test = np.argmax(y_test, axis=1)

      d = {'pred': Y_pred, 'true': y_test}
      df = pd.DataFrame(data=d)
      df['Misclassified'] = df.pred != df.true

      if showClass !='all':
        if showClass in classList:
          showClassIndex = classList.index(showClass)
        else:
          raise ValueError('No `{}` found in `"class list"`.'.format(showClass))
        if order == 'random':
          imageIndex = random.sample(df.index[(df.pred==showClassIndex)&(df.Misclassified)].tolist(), n)
        elif order == 'first':
          imageIndex = df.index[(df.pred==showClassIndex)&(df.Misclassified)].tolist()[:n]
        elif order == 'last':
            imageIndex = df.index[(df.pred==showClassIndex)&(df.Misclassified)].tolist()[-n:]
        else:
          raise ValueError('`Order` must be one of `"first"`, `"last"` or `"random"`.')
      else:
        if order == 'random':
            imageIndex = random.sample(df.index[df.Misclassified].tolist(), n)
        elif order == 'first':
            imageIndex = df.index[df.Misclassified].tolist()[:n]
        elif order == 'last':
            imageIndex = df.index[df.Misclassified].tolist()[-n:]
        else:
            raise ValueError('`Order` must be one of `"first"`, `"last"` or `"random"`.')

      rows = (n + columns - 1) / columns
      fig = plt.figure(figsize=figsize)

      for i in range(n):
          img = x_test[imageIndex[i]]
          ax = fig.add_subplot(rows, columns, i + 1, xticks=[], yticks=[])
          if classList:
              ax.set_title('True:%s, Predict:%s' % (
              classList[df.iloc[imageIndex[i]].true], classList[df.iloc[imageIndex[i]].pred]))
          if np.max(img) < 20:  # threshold of 20
              min_val = np.min(img)
              max_val = np.max(img)
              img_unnormalized = (img - min_val) / (max_val - min_val)
              plt.imshow(img_unnormalized)
          else:
              plt.imshow(img)

      if tightLayout:
          fig.tight_layout()
      plt.axis('off')