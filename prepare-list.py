import pandas as pd
import pickle

men_classes = pd.read_csv(".\\images_labelling.csv")

men_list = {}

for i in range(len(men_classes)):
  if  not men_classes["label"][i] in men_list:
    men_list[men_classes['label'][i]] = men_classes["class_"][i]
    
pickle_out = open("man-list.pickle","wb")
pickle.dump(men_list, pickle_out)
pickle_out.close()
print(men_list)