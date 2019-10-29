#batch iterator class
import logging
from data_generation import rotate_triangle, translate_triangle, crop_image_centre, crop_array_centre, angle_to_vector, angle_to_class_matrix
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import data_handling as dh


default_data_params = {
    'batch_size': 32,
    'real_or_synthetic': 'synthetic',
    'fov':200,

    #data augmentation: rotate per epoch, or once, or False
    'rotate':False,
    'noise':False,

    'val_fraction': 0.01,
    'test_fraction': 0.01,
    
    #labelled_data type: angle, vector, nxm classification
    'label_type':'angle',

    
    #section number
    'json': None,
    'histo':None,
    'shuffle':True,
    'seed':False,
    
    #synthetic data parameters
    'number_of_examples':5000,
    'jitter':20,
    
}

from tensorflow.keras.preprocessing.image import Iterator
from scipy import ndimage


class HistoBatchIterator(Iterator):
    def __init__(self, params, mode):
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.info('Initializing...')
        default_params = default_data_params.copy()
        for name in default_params.keys():
            setattr(self, name, params.get(name, default_params[name]))
            self.log.info('Setting {} to {:200}'.format(name, '{}'.format(getattr(self, name))))
        self.mode=mode   
        #returns larger images, to be rotated and cropped as necessary
        self.load_dataset()
        #initial rotations. Can be repeated per epoch later

 
        N=len(self.base_images)

        super(HistoBatchIterator, self).__init__(N, self.batch_size, self.shuffle, self.seed)
        
        
    def _get_batches_of_transformed_samples(self, index_array):
        # iterate through the current batch
        batch_x, batch_y = self.rotate_images(self.base_images[index_array],self.base_angles[index_array], self.fov)
        #TODO output batch_y according to self.label_type
        batch_y = self.format_angles(batch_y, self.label_type)
        
        #add any noise or data things here
        if self.noise:
            batch_x +=np.random.randn(len(batch_x),self.fov, self.fov,1)*self.noise
        batch_x = np.clip(batch_x,0,255)
        batch_x = batch_x/255
        #for torch swap axes
        batch_x = np.repeat(batch_x,3,axis=3)
        batch_x = np.swapaxes(batch_x,1,3)
        batch_x = np.swapaxes(batch_x,2,3)
        return batch_x, batch_y
    
            
    def next(self):
        with self.lock:
            # get input data index and size of the current batch
            index_array = next(self.index_generator)
        # create array to hold the images
        return self._get_batches_of_transformed_samples(index_array)
    
    def format_angles(self, batch_y, label_type):
        if label_type=='angle':
            return batch_y
        elif label_type =='vector':
            #TODO create function and reverse function
            batch_y=angle_to_vector(np.deg2rad(batch_y))
        elif label_type == 'classification':
            #TODO create function and reverse function ??tf??
            batch_y = angle_to_class_matrix(batch_y, self.class_n,self.class_m)
        return batch_y
    
    
    
    def rotate_images(self,base_images,base_angles, target_box_size=200,rotate_range=360):
        number_of_examples=len(base_images)
        images=np.zeros((number_of_examples,target_box_size,target_box_size,1))
        angles=np.random.randint(0,rotate_range, number_of_examples)
        for k,angle in enumerate(angles):
            base=base_images[k].copy()
            rotated = ndimage.rotate(base, 360-angle, reshape=False)
            image=crop_array_centre(rotated,target_box_size)
            images[k]=np.array(image).reshape(target_box_size,target_box_size,1)
        rotated_angles=base_angles+angles
        rotated_angles[rotated_angles>360]-=360
        #clip to 180
        #rotated_angles[rotated_angles>180]-=180
        
        return images, rotated_angles
        
    
        
    def load_dataset(self):
        if self.real_or_synthetic =='synthetic':
            if self.mode =='val':
                self.number_of_examples=np.round(self.number_of_examples*self.val_fraction).astype(int)
            elif self.mode =='test':
                self.number_of_examples=np.round(self.number_of_examples*self.test_fraction).astype(int)
            self.generate_synthetic_dataset(target_box_size=self.fov, 
                                           number_of_examples=self.number_of_examples,
                                           jitter=self.jitter)
        elif self.real_or_synthetic =='real':
            #TODO generate real dataset functions
            #Consider loading inside the batch iterator
            self.generate_real_dataset()
            
    def generate_real_dataset(self):
        
        angles_image, mask = dh.load_and_process_histology(self.json, self.histo)
        self.generate_n_examples(self.histo, angles_image, mask, target_box_size=self.fov, number_of_examples=self.number_of_examples)

    def generate_n_examples(self,image_file_name, angles_image, mask, target_box_size=200, number_of_examples=1000,precision=1):
        """generate matched pairs of tissue and labelled angle
        can choose to rotate to vertical. 
        In this case box size will be used to calculate crop size prior to rotation and recropping"""
        box_size=np.ceil(np.sqrt(2*target_box_size**2)).astype(int)
        im=Image.open(image_file_name).convert('L')
        w,h=im.size
        self.base_images=np.zeros((number_of_examples,box_size,box_size))
        self.base_angles=np.zeros(number_of_examples)
        k=0
        while k < number_of_examples:
            y=np.random.randint(0,h-box_size)
            x=np.random.randint(0,w-box_size)

            #mask check
            if np.mean(mask[y:y+box_size,x:x+box_size])<0.8:
                pass
            else:
                bounding_box=[x,y,x+box_size,y+box_size]
                self.base_images[k] = dh.generate_tissue_box(im,bounding_box)
                self.base_angles[k]= dh.calculate_box_angle(angles_image, bounding_box)
                k+=1
    
    def generate_synthetic_dataset(self, target_box_size=200, number_of_examples=1000, precision=1, 
                     n_triangles=40,jitter=3, translate_size=150):
        """create triangular image of speicifed size and rotate + add some speckled noise
        n_triangles - number of triangles to plot
        jitter = degree of jitter of triangle rotations
        noise - random noise
        precision - return angles binned to specified precision eg 0-35 for precision 10"""
        #calculate larger base image so that when rotated, target size is still inside
        box_size=np.ceil(np.sqrt(2*target_box_size**2)).astype(int)
        self.base_images=np.zeros((number_of_examples,box_size,box_size))
        for example in range(number_of_examples):
            im=Image.new('L',(box_size,box_size),0)
            com=np.array((box_size/2,box_size/2))
            rand_n_triangles=np.random.randint(1,n_triangles)
            for t in range(rand_n_triangles):
            #triangle pyramidal neuron
                tri_size=np.random.randint(10,20)
                triangle=np.round(np.array(((com+(0,-tri_size/1.5)),
                                           (com+(tri_size/4,tri_size/2)),
                                           (com+(-tri_size/4,tri_size/2))))).astype(int)
                triangle=tuple(map(tuple,triangle))
                jitter_angle=np.random.randint(-jitter,jitter)
                new_r=rotate_triangle(com, triangle, jitter_angle)
                triangle= np.round(translate_triangle(new_r, translate_size)).astype(int)
                intensity=np.random.randint(140,255)

                ImageDraw.Draw(im).polygon(tuple(map(tuple,triangle)),  fill=intensity, outline =intensity)
            im = im.filter(ImageFilter.GaussianBlur(2))
            self.base_images[example]=np.array(im)
        self.base_angles=np.zeros(number_of_examples)
        return 
    
    
    