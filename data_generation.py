import numpy as np
def crop_image_centre(image, box_size):
    """crop the central square from an image, eg if rotated"""
    width, height = image.size
    left = (width-box_size)/2
    top = (height - box_size)/2
    right = (width+box_size)/2
    bottom = (height+box_size)/2
    return image.crop((left,top,right, bottom))


def crop_array_centre(image, box_size):
    """crop the central square from an image, eg if rotated"""
    width, height = image.shape
    left = np.round((width-box_size)/2).astype(int)
    top = np.round((height - box_size)/2).astype(int)
    right = np.round((width+box_size)/2).astype(int)
    bottom = np.round((height+box_size)/2).astype(int)
    return image[top:bottom,left:right]
    
def generate_n_examples(image_file_name,angles_image,mask, target_box_size, number_of_examples,rotate=True,precision=1):
    """generate matched pairs of tissue and labelled angle
    can choose to rotate to vertical. 
    In this case box size will be used to calculate crop size prior to rotation and recropping"""
    if rotate:
        box_size=np.ceil(np.sqrt(2*target_box_size**2)).astype(int)
    else:
        box_size=target_box_size
    im=Image.open(image_file_name).convert('L')
    w,h=im.size
    images=[]
    angles=[]
    while len(images) < number_of_examples:
        y=np.random.randint(0,h-box_size)
        x=np.random.randint(0,w-box_size)
        
        #mask check
        if np.mean(mask[y:y+box_size,x:x+box_size])<0.6:
            pass
        else:
            bounding_box=[x,y,x+box_size,y+box_size]
            imc = dh.generate_tissue_box(im,bounding_box)
            angle= dh.calculate_box_angle(angles_image, bounding_box)
            images.append(imc)
            angles.append(angle)
    if rotate:
        print('rotating')
        rotated_images=np.zeros((number_of_examples,target_box_size,target_box_size,1))
        rotation_angles=np.random.randint(0,360, number_of_examples)
        for k, rotation_angle in enumerate(rotation_angles):
            rotated = images[k].rotate(rotation_angle)
            cropped=crop_image_centre(rotated,target_box_size)
            cropped=gaussian_filter(cropped,2)

            rotated_images[k]=np.array(cropped).reshape(target_box_size,target_box_size,1)
            
        images=rotated_images
        angles=(angles-rotation_angles)%360
    orig_images=np.zeros((number_of_examples,target_box_size,target_box_size,1))
    for k,image in enumerate(images):
        orig_images[k]=np.array(image).reshape(target_box_size,target_box_size,1)
    #orig_images=(orig_images<110).astype(int)
    orig_images=orig_images/255.
    return orig_images, np.array(angles)

import math

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def rotate_triangle(origin, triangle, angle):
    """rotate 3 points around specified origin"""
    angle=2*math.pi*angle/360
    return (rotate(origin, triangle[0],angle),rotate(origin, triangle[1],angle),rotate(origin, triangle[2],angle))

def translate_triangle(triangle, box_size):
    """translate triangle from centre by random amount """
    translation_vector=np.random.uniform(-box_size/2,box_size/2,2)
    triangle = triangle+translation_vector
    return triangle


    

def simulated_images(target_box_size, number_of_examples, rotate_range=360,noise=False,precision=1, 
                     n_triangles=1,jitter=3):
    """create triangular image of speicifed size and rotate + add some speckled noise
    n_triangles - number of triangles to plot
    jitter = degree of jitter of triangle rotations
    noise - random noise
    precision - return angles binned to specified precision eg 0-35 for precision 10"""
    
    box_size=np.ceil(np.sqrt(2*target_box_size**2)).astype(int)
    base_images=[]
    for example in range(number_of_examples):
        im=Image.new('L',(box_size,box_size),0)
        com=np.array((box_size/2,box_size/2))
        rand_n_triangles=np.random.randint(1,n_triangles)
        for t in range(rand_n_triangles):
            
#        if n_triangles==1:
        #triangle pyramidal neuron
            tri_size=np.random.randint(10,20)
            #TODO triangle needs to be centred
            
            triangle=np.round(np.array(((com+(0,-tri_size/1.5)),
                                       (com+(tri_size/4,tri_size/2)),
                                       (com+(-tri_size/4,tri_size/2))))).astype(int)
            triangle=tuple(map(tuple,triangle))
            
            translate_size=150
            jitter_angle=np.random.randint(-jitter,jitter)
            new_r=rotate_triangle(com, triangle, jitter_angle)
            triangle= np.round(translate_triangle(new_r, translate_size)).astype(int)
            intensity=np.random.randint(140,255)

            ImageDraw.Draw(im).polygon(tuple(map(tuple,triangle)),  fill=intensity, outline =intensity)
        base_images.append(im)
        


    images=np.zeros((number_of_examples,target_box_size,target_box_size,1))
    if rotate_range:
        angles=np.random.randint(0,rotate_range, number_of_examples)
    else:
        angles=np.zeros(number_of_examples)
    for k,angle in enumerate(angles):
        rotated=base_images[k].rotate(360-angle)
        image=crop_image_centre(rotated,target_box_size)
        
        image=np.array(image).reshape(target_box_size,target_box_size,1)
        if noise:
            image=image+np.abs(np.random.randn(target_box_size, target_box_size,1))*noise
            image[image>255]=255
        #images[k]=image
        #smooth
        image=gaussian_filter(image,2)
        #binarize
        #image=(image>140).astype(int)
        images[k]=image /255.
    return images, bin_angles(angles,precision)


#TODO bin_angles with arbitrary bins, return classification targets nxm matrix n directions, m unique orientations
def angle_to_vector(angles):
    """convert angle to vectors"""
    return np.vstack((np.sin(angles), np.cos(angles))).T

def angle_to_class_matrix(angles, n,m):
    """define angles using on-hot coding of n angles, m times over"""
    bin_spacing = 360/n
    shift = bin_spacing/m
    angle_matrix=np.zeros((m,len(angles)))
    print(shift)
    for encoding in range(m):
        transform_angles=(angles+shift*encoding)%360
        angle_matrix[encoding] = bin_angles(transform_angles, bin_spacing)
    return angle_matrix
        
def bin_angles(angles,precision):
    """bin labelled angles to degree of specified accuracy"""
    binned=np.floor(angles/precision).astype(int)
    #set max to zero
    return binned