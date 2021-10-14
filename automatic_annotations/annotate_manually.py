from shutil import copyfile
import cv2 as cv
import numpy as np  
import os


ML_PATH = r'H:\Konrad\tcm_scan\20210621_092043_data'

# ML_PATH
ML_PATH_annotations =  ML_PATH + r'\annotations\xmls' 
ML_PATH_images =  ML_PATH + r'\images' 
ML_PATH_images_F =  ML_PATH + r'\images_F' 



draw = False
a,b = -1,-1

def create_annotation(img,image_name,xmin,ymin,xmax,ymax):
    path = ML_PATH_images + '\\' + image_name
    base_name = image_name[:image_name.rfind('.')]
    xml_name = ML_PATH_annotations +'\\' + base_name + '.xml'
    folder = ML_PATH[ML_PATH.rfind('\\')+1:]
   
    anntoation='''
<annotation>
	<folder>{}</folder>
	<filename>{}</filename>
	<path>{}</path>
	<source>
		<database>TCM_database</database>
	</source>
	<size>
		<width>{}</width>
		<height>{}</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>tooth</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>{}</xmin>
			<ymin>{}</ymin>
			<xmax>{}</xmax>
			<ymax>{}</ymax>
		</bndbox>
	</object>
</annotation>
'''.format(folder,image_name,path,img.shape[1],img.shape[0],xmin,ymin,xmax,ymax)   
    print(xml_name)
    #print(anntoation)
    f = open(xml_name, "w")
    f.write(anntoation)
    f.close()
    print(anntoation)


# Creating mouse callback function  
def draw_rectangle(event,x,y,flags,param):  
    global a,b,draw
    global xmin,ymin,xmax,ymax 
    
    if(event == cv.EVENT_LBUTTONDOWN):  
        draw = True  
        a,b = x,y 
        xmin,ymin = a,b
        cv.imshow('image',param)
        
    elif (event == cv.EVENT_MOUSEMOVE):     
        if draw == True:
            drawing = img.copy()
            xmax,ymax = x,y
            cv.putText(drawing, text, (100,100), 5, 5, (0,255,0), 2, cv.LINE_AA)      
            cv.rectangle(drawing,(a,b),(x,y),(0,255,0),3)  
            cv.imshow('image',drawing)
            
    elif(event == cv.EVENT_LBUTTONUP):  
        draw = False 
        


files = list(os.listdir(ML_PATH_images_F))
for i,image_name in enumerate(files):
   
    img = cv.imread(os.path.join(ML_PATH_images_F,image_name))
    cv.namedWindow('image',cv.WINDOW_FREERATIO)  
    
    drawing = img.copy()
    text = "Image {}/{}".format(i,len(files))
    cv.putText(drawing, text, (100,100), 5, 5, (0,255,0), 2, cv.LINE_AA)
    cv.imshow('image',drawing)
    cv.setMouseCallback('image',draw_rectangle,drawing)
   
    while(1):  
        cv.resizeWindow('image', 1200, 900) 
        if cv.waitKey(20) & 0xFF == 32:  
            break  
    create_annotation(img,image_name,xmin,ymin,xmax,ymax)
    print(ML_PATH_images_F + '\\'+image_name,'---------->',ML_PATH_images + '\\'+image_name)
    cv.imwrite(ML_PATH_images + '\\'+ image_name,img)
    os.remove(ML_PATH_images_F + '\\'+ image_name)
    cv.destroyAllWindows()  