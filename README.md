
# Documentation
## utils
### Class Gallery
You can instantiate a Gallery via:

    from Jnana.utils import Gallery
    gallery=Gallery(model,x_train,y_train, size=50000, num_classes=10)

    

#### Methods

***show***

    show(self, n=20, classList=None, showClass='all', columns=6, figsize=(16, 16), order='random', tightLayout=True):
Displays an image gallery  
**Arguments**
 - n: [int] Number of images 
 - classList: *[list]* Label/Class names for dataset
 - showClass: *[string]*: Name of the class that needs to be displayed
 	 - *'all'*: displays all the classes
	 - *'< classname >'*: displays this particular class only
 - columns: *[int]* number of columns for image gallery
 - figsize: *[tuple(int,int)]* size of each image in image gallery
 - order: *[string]* sequence of image selection from the dataset
	 - *'first'*: selects first n images from dataset
	 - *'last'*: selects last n images from dataset
	 - *'random'*: selects random n images from dataset
 - tightLayout: *[bool]* fight layout cleanly
 
 **Raises**  
 *ValueError*: In case of mismatch between the provided value for *order*.  
 *ValueError*: If *showClass* is provided as a input parameter and *classList* is not  
 *ValueError*: If *showClass* is not in *classList*

***showMisclassified***

    showMisclassified(self, x_test=None, y_test=None, n=20, classList=None, showClass='all', columns=6, figsize=(16, 16), order='random', tightLayout=True)
Displays an image gallery for mis-classified images for a trained model  
**Arguments**
 - x_test: *[numpy.ndarray]* testing data
 - y_test: *[numpy.ndarray]* label/target data
 - n: *[int]* Number of images 
 - classList: *[list]* Label/Class names for dataset
 - showClass: *[string]*: Name of the class that needs to be displayed
 	 - *'all'*: displays all the classes
	 - *'< classname >'*: displays this particular class only
 - columns: *[int]* number of columns for image gallery
 - figsize: *[tuple(int,int)]* size of each image in image gallery
 - order: *[string]* sequence of image selection from the dataset
	 - *'first'*: selects first n images from dataset
	 - *'last'*: selects last n images from dataset
	 - *'random'*: selects random n images from dataset
 - tightLayout: *[bool]* fight layout cleanly  
 
 **Raises**  
 *ValueError*: In case of mismatch between the provided value for *order*.  
 *ValueError*: If *showClass* is provided as a input parameter and *classList* is not  
 *ValueError*: If *showClass* is not in *classList*
