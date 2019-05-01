import numpy
import cv2
#NOTE: iam using an image dataset that i downloaded from kaggle.com for training and testing
#NOTE:iam restricting the svm to work with numbers from 1 to 5 thus why iam using (1,6) in the proceeding code

#array to hold training data
tData=[]
#array of labels to associate with training data
labels=[]

svm=cv2.ml.SVM_create()
kaze=cv2.KAZE_create()
customExtractor=cv2.BOWImgDescriptorExtractor(kaze,cv2.FlannBasedMatcher(dict(algorithm=1,trees=5),{}))
trainer=cv2.BOWKMeansTrainer(40)
for j in range(1,6):
    for i in range(1,21):
        k,d=kaze.detectAndCompute(cv2.imread("%d/img_%d.jpg"%(j,i),0),None)
        trainer.add(d)

voc=trainer.cluster()
customExtractor.setVocabulary(voc)

for j in range(1,6):
    for i in range(1,21):
        gray=cv2.imread("%d/img_%d.jpg"%(j,i),0)
        d=customExtractor.compute(gray,kaze.detect(gray))
        tData.extend(d)
        labels.append(j)

svm.train(numpy.asarray(tData),cv2.ml.ROW_SAMPLE,numpy.asarray(labels))

#read photo and predict
img=cv2.imread("4/img_78.jpg")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
des=customExtractor.compute(gray,kaze.detect(gray))
p=svm.predict(des)

for i in range(1,6):
    if(p[1]==i):
        print(i)
        #cv2.putText(img,str(i),(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,200),3)
        cv2.imshow("number",img)
        break
