# This code creates a repository named Satellite_Image_Processing_Airport and provides us normalised images, histogram equalised images, smoothed classification into 3 areas (Slum, non-slum (buildings or structured area) and anything other than that.
# It also provides us a labeled data into 3 regions according to the smoothed classification images                                                     
import os
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import cv2
from skimage import exposure, morphology, filters
from sklearn.preprocessing import MinMaxScaler
from scipy import ndimage
from tqdm import tqdm
import time

class SatelliteImageProcessor:
    def __init__(self, input_dir, output_dir, region=(2700, 3000, 700, 1000)):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.region = region
        
        self.create_output_directories()

    def create_output_directories(self):
        output_subdirs = [
            'false_color_composites',
            'normalized_images',
            'histogram_equalized',
            'smoothed_classification',
            'labeled_classification'
        ]
        
        self.output_subdirs = {}
        for subdir in output_subdirs:
            path = os.path.join(self.output_dir, subdir)
            os.makedirs(path, exist_ok=True)
            self.output_subdirs[subdir] = path

    def load_multiband_image(self):
        tiff_files = sorted([f for f in os.listdir(self.input_dir) if f.endswith('.tif')])
        
        first_image = tiff.imread(os.path.join(self.input_dir, tiff_files[0]))
        mum_mult = np.zeros((first_image.shape[0], first_image.shape[1], len(tiff_files)))
        
        for i, filename in tqdm(enumerate(tiff_files), total=len(tiff_files), desc="Loading Multiband Image"):
            mum_mult[:,:,i] = tiff.imread(os.path.join(self.input_dir, filename))
            time.sleep(0.1)
        
        return mum_mult

    def normalize_image(self, mum_mult):
        with tqdm(total=100, desc="Normalizing Image") as pbar:
            mum_mult_flat = np.reshape(mum_mult, (mum_mult.shape[0] * mum_mult.shape[1], mum_mult.shape[2]))
            pbar.update(30)
            
            scaler = MinMaxScaler()
            mum_norm = scaler.fit_transform(mum_mult_flat)
            pbar.update(30)
            
            normalized = np.reshape(mum_norm, (mum_mult.shape[0], mum_mult.shape[1], mum_mult.shape[2]))
            zoomed_normalized = normalized[self.region[0]:self.region[1], self.region[2]:self.region[3], :3]
            pbar.update(20)
            
            plt.figure(figsize=(15, 15))
            plt.imshow(zoomed_normalized)
            plt.title('Normalized Airport Region')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_subdirs['normalized_images'], 'normalized_airport.png'))
            plt.close()
            pbar.update(20)
        
        return normalized

    def histogram_equalization(self, mum_mult_norm):
        with tqdm(total=100, desc="Histogram Equalization") as pbar:
            mumbai_hist_equ = exposure.equalize_hist(mum_mult_norm)
            pbar.update(50)
            
            hist_equ = (mumbai_hist_equ * 255).astype('uint8')
            zoomed_hist_equ = hist_equ[self.region[0]:self.region[1], self.region[2]:self.region[3], :3]
            pbar.update(30)
            
            plt.figure(figsize=(15, 15))
            plt.imshow(zoomed_hist_equ)
            plt.title('Histogram Equalized Airport Region')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_subdirs['histogram_equalized'], 'hist_equ_airport.png'))
            plt.close()
            pbar.update(20)
        
        return hist_equ

    def save_false_color_images(self, mumbai_hist_equ):
        composites = {
            'RGB': (2, 1, 0),   
            'NIR_Red_Green': (3, 2, 1), 
            'SWIR_NIR_Red': (4, 3, 2)    
        }
        
        with tqdm(total=100, desc="Creating False Color Composites") as pbar:
            for name, bands in composites.items():
                start_row, end_row = self.region[0], self.region[1]
                start_col, end_col = self.region[2], self.region[3]
                
                composite = mumbai_hist_equ[start_row:end_row, start_col:end_col, bands]
                
                plt.figure(figsize=(15, 15))
                plt.imshow(composite)
                plt.title(f'{name} Composite - Airport Region')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_subdirs['false_color_composites'], f'{name}_composite.png'))
                plt.close()
                
                pbar.update(100 // len(composites))

    def three_class_classification(self, mumbai_hist_equ):
        with tqdm(total=100, desc="Three-Class Classification") as pbar:
            start_row, end_row = self.region[0], self.region[1]
            start_col, end_col = self.region[2], self.region[3]
            temp = mumbai_hist_equ[start_row:end_row, start_col:end_col, :]
            pbar.update(20)
            
            temp_gray = np.mean(temp, axis=2)
            pbar.update(20)
            
            otsu_threshold_low = filters.threshold_otsu(temp_gray)
            otsu_threshold_high = np.percentile(temp_gray, 75)
            pbar.update(20)
            
            classification_mask = np.zeros_like(temp_gray, dtype=np.uint8)
            classification_mask[temp_gray < otsu_threshold_low] = 1  
            classification_mask[temp_gray >= otsu_threshold_high] = 2  
            pbar.update(20)
            
            plt.figure(figsize=(15, 15))
            plt.imshow(classification_mask, cmap='viridis')
            plt.title('Three-Class Classification')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_subdirs['smoothed_classification'], 'three_class_classification.png'))
            plt.close()
            pbar.update(10)
        
        return classification_mask

    def label_three_class_classification(self, classification_mask):
        with tqdm(total=100, desc="Labeling Classification") as pbar:
            labeled_image = np.zeros((classification_mask.shape[0], classification_mask.shape[1], 3), dtype=np.uint8)
            
            labeled_image[classification_mask == 1] = [255, 69, 0]   
            labeled_image[classification_mask == 2] = [34, 139, 34]  
            labeled_image[classification_mask == 0] = [65, 105, 225]
            pbar.update(50)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            color = (255, 255, 255)
            thickness = 2
            
            labels = [
                f"Region: {self.region}",
                "Orange: Unplanned Areas (Slums)",
                "Green: Planned Areas (Buildings)",
                "Blue: Other Areas"
            ]
            
            for i, text in enumerate(labels):
                cv2.putText(labeled_image, text, (10, 30 + i*30), 
                            font, font_scale, color, thickness, cv2.LINE_AA)
            pbar.update(30)
            
            cv2.imwrite(os.path.join(self.output_subdirs['labeled_classification'], 'labeled_three_class_classification.png'), labeled_image)
            pbar.update(20)
        
        return labeled_image

    def process(self):
        with tqdm(total=100, desc="Overall Processing") as overall_pbar:
            mum_mult = self.load_multiband_image()
            overall_pbar.update(20)
            
            mum_mult_norm = self.normalize_image(mum_mult)
            overall_pbar.update(20)
            
            mumbai_hist_equ = self.histogram_equalization(mum_mult_norm)
            overall_pbar.update(20)
            
            self.save_false_color_images(mumbai_hist_equ)
            overall_pbar.update(10)
            
            classification_mask = self.three_class_classification(mumbai_hist_equ)
            overall_pbar.update(20)
            
            self.label_three_class_classification(classification_mask)
            overall_pbar.update(10)

def main():
    input_dir = 'D:/MPR' #change according to your directory
    output_dir = 'D:/MPR/Satellite_Image_Processing_Airport' #change name and destination according to your needs
    
    processor = SatelliteImageProcessor(input_dir, output_dir)
    processor.process()

if __name__ == '__main__':
    main()
