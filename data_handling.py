import numpy as np
import json
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from matplotlib.path import Path




#load json
def load_json_annotations(json_file):
    with open(json_file,'r') as f:
        lines=json.load(f)
    #import column data
    drawn_columns=[]
    for shape in lines.get('shapes'):
        if shape['shape_type'] == 'linestrip':
            drawn_columns.append(np.round(shape['points']).astype(int))
        elif shape['shape_type']=='polygon':
            cortex=shape['points']

    cortex=np.round(cortex).astype(int)
    return drawn_columns, cortex

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    linalg=np.linalg.norm(vector)
    if linalg ==0:
        return [0,0]
    else:
        return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    dot = v1_u[0]*v2_u[0] + v1_u[1]*v2_u[1]
    det = v1_u[0]*v2_u[1] - v1_u[1] * v2_u[0]
    return np.arctan2(det, dot)


def mask_image(png_file, cortex_mask_polygon):
    """load image, mask out cortex annotation"""
    #load image file and convert to rgb
    w,h = Image.open(png_file).size
    img= Image.new('L',(w,h),0)
    ImageDraw.Draw(img).polygon(tuple(map(tuple,cortex_mask_polygon)),  fill=1, outline =1)
    mask=np.array(img)
    return mask

def columns_to_angles_image(columns,mask):
    """calculate angles from lines and plot onto image"""
    #create blank image
    angles_image = Image.new('L',(mask.shape[1],mask.shape[0]),0)
    #needed to ensure angles of 0 are recognised and dilated
    angles_mask = Image.new('L',(mask.shape[1],mask.shape[0]),0)
    for column in columns:
        for p in range(len(column)-1):
            v1=column[p]-column[p+1]
            #calculate angle and convert to degrees (0-360)
            angle=np.degrees(angle_between([0,1],v1))
            #deal with NANs
            if angle <-360:
                angle=0
            #set negatives to >180 as images need positive values
            elif angle < 0:
                angle=360+angle
            #rescale to fit inside uint8
            angle=255.*angle/360.
            angle=np.round(angle).astype(int)
            ImageDraw.Draw(angles_image).line([tuple(column[p+1]),tuple(column[p])],fill=int(angle),width=20)
            ImageDraw.Draw(angles_mask).line([tuple(column[p+1]),tuple(column[p])],fill=1,width=20)

    return angles_image, angles_mask

def dilate_values_inside_cortex(angles_image,angles_mask, cortex_mask):
    """dilate values inside a cortical mask to fill cortex with nearest value"""
    # dilate
    import cv2
    from scipy.ndimage import gaussian_filter
    #establish expansion kernel size
    kernel=np.ones((5,5),np.uint8)
    new_image=np.array(angles_image)
    old_mask=np.array([1])
    angles_mask=np.array(angles_mask)
    while np.sum(angles_mask)!=np.sum(old_mask):
        old_mask=angles_mask.copy()
        
        new_image[angles_mask==0]+=cv2.dilate(new_image,kernel, iterations = 5)[angles_mask==0]
        angles_mask=cv2.dilate(angles_mask,kernel,iterations=5)
        angles_mask[cortex_mask==0]=0
        new_image[cortex_mask==0]=0
    new_image[cortex_mask==0]=0
        
    #convert back to angles
    angles_image=360*new_image.astype(np.uint32)/255.0
    return angles_image


def generate_tissue_box(image_file,bounding_box):
    """cut small piece of tissue from image opened in PIL"""
    imc = image_file.crop(bounding_box)
    return imc


def load_and_process_histology(json_file, image_file):
    """load in annotated histology and expand annotations to make angle image
    json_file - containing annotations
    image_file - 1micron histo image"""
    print('loading json')
    drawn_columns, cortex = load_json_annotations(json_file)
    print('generating mask')
    mask = mask_image(image_file,cortex)
    print('drawing columns')
    angles_image, angles_mask=columns_to_angles_image(drawn_columns,mask)
    print('dilating columns')
    angles_image=dilate_values_inside_cortex(angles_image,angles_mask,mask)
    return angles_image, mask
        
        
def calculate_box_angle(angles_image, bounding_box):
    #stopped median as overweights edge vertices
    #median_angle=np.median(angles_image[bounding_box[1]:bounding_box[3],bounding_box[0]:bounding_box[2]])
    #take central value
    angle = angles_image[np.mean([bounding_box[1],bounding_box[3]]).astype(int),np.mean([bounding_box[0],bounding_box[2]]).astype(int)]
    return angle

def generate_n_examples(image_file_name,angles_image,mask, box_size, number_of_examples):
    """generate matched pairs of tissue and labelled angle"""
    im=Image.open(image_file_name)
    w,h=im.size
    pairs=[]
    while len(pairs) < number_of_examples:
        y=np.random.randint(0,h-box_size)
        x=np.random.randint(0,w-box_size)
        
        #mask check
        if np.mean(mask[y:y+box_size,x:x+box_size])<0.6:
            pass
        else:
            bounding_box=[x,y,x+box_size,y+box_size]
            imc = generate_tissue_box(im,bounding_box)
            angle= calculate_box_angle(angles_image, bounding_box)
            pairs.append([imc,angle])
    return pairs

def generate_arrow_figure(image_file_name, angles_image, mask, box_size):
    im= Image.open(image_file_name)
    w,h=im.size
    x_s=np.arange(0,w,box_size)
    y_s=np.arange(0,h,box_size)
    fig = plt.figure(figsize=(10,10))
    plt.imshow(im,cmap='gray')
    for x in x_s:
        for y in y_s:
            bounding_box=[x,y,x+box_size,y+box_size]
            if np.mean(mask[y:y+box_size,x:x+box_size])>0.6:
                angle= calculate_box_angle(angles_image, bounding_box)
                liney,linex = np.sin(np.deg2rad(90+angle)),np.cos(np.deg2rad(90+angle))
                plt.plot([x+-linex*box_size/3.+box_size/2.,x+linex*box_size/3+box_size/2.],[y+-liney*box_size/3+box_size/2.,y+liney*box_size/3+box_size/2.],'r',linewidth=3)
    return fig