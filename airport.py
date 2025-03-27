#In this code, we are focusing on the region near the airport (Mumbai area)
#This will save all the images in the output directory after performing normalisation (on all 10 images), then false color composites and then it will zoom and give outputs too
#We have added 2 levels of zoom on the images and both of them can be seen in the output directory 

import numpy as np
import tifffile as tiff
import os
from sklearn.preprocessing import MinMaxScaler
from skimage import exposure
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import rasterio
import warnings
warnings.filterwarnings('ignore')

def stack_bands_with_georeference(directory):
    band_names = [
        'band2_blue',
        'band3_green',
        'band4_red',
        'band5_rededge1',
        'band6_rededge2',
        'band7_rededge3',
        'band8_nir',
        'band8A_nir',
        'band11_swir',
        'band12_swir'
    ]
    
    first_band_path = os.path.join(directory, f'{band_names[0]}.tif')
    with rasterio.open(first_band_path) as src:
        metadata = src.profile
        height = src.height
        width = src.width
    
    multispectral_img = np.zeros((height, width, len(band_names)), dtype=np.float32)
    
    for idx, band_name in enumerate(band_names):
        file_path = os.path.join(directory, f'{band_name}.tif')
        try:
            with rasterio.open(file_path) as src:
                multispectral_img[:,:,idx] = src.read(1)
            print(f'Successfully loaded {band_name} with shape {multispectral_img[:,:,idx].shape}')
        except Exception as e:
            print(f'Error loading {band_name}: {str(e)}')
    
    return multispectral_img, metadata, band_names

def process_multispectral_data(multispectral_img):
    orig_shape = multispectral_img.shape
    flat_img = np.reshape(multispectral_img, (orig_shape[0] * orig_shape[1], orig_shape[2]))
    print(f"Flattened shape: {flat_img.shape}")
    
    scaler = MinMaxScaler()
    norm_flat = scaler.fit_transform(flat_img)
    print(f"Normalized flat shape: {norm_flat.shape}")
    
    norm_img = np.reshape(norm_flat, orig_shape)
    print(f"Normalized image shape: {norm_img.shape}")
    
    hist_eq = exposure.equalize_hist(norm_img)
    hist_eq = (hist_eq * 255).astype('uint8')
    print(f"Final image shape: {hist_eq.shape}")
    
    return hist_eq

def create_rgb_image(img, bands=(3,2,1)):
    rgb = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for i in range(3):
        if bands[i] < img.shape[2]:
            rgb[:,:,i] = img[:,:,bands[i]]
    return rgb

def visualize_processed_data(processed_img, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    
    rgb_image = create_rgb_image(processed_img, (3,2,1))
    plt.figure(figsize=(25, 25))
    plt.imshow(rgb_image)
    plt.title('RGB Composite (Red-Green-Blue)')
    plt.axis('off')
    plt.savefig(os.path.join(output_directory, 'rgb_composite.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    zoomed_rgb1 = create_rgb_image(processed_img[2500:3000, 500:1000], (3,2,1))  #LEVEL - 1 Zoom
    plt.figure(figsize=(25, 25))
    plt.imshow(zoomed_rgb1)
    plt.title('Zoomed RGB Composite (First Level)')
    plt.axis('off')
    plt.savefig(os.path.join(output_directory, 'zoomed_rgb_level1.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    
    zoomed_rgb2 = create_rgb_image(processed_img[2700:3000, 700:1000], (3,2,1))   #LEVEL - 2 Zoom
    plt.figure(figsize=(15, 15))
    plt.imshow(zoomed_rgb2)
    plt.title('Zoomed RGB Composite (Second Level)')
    plt.axis('off')
    plt.savefig(os.path.join(output_directory, 'zoomed_rgb_level2.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    
    false_color_combinations = [
        ((7,3,2), 'NIR-Green-Blue'),
        ((7,4,3), 'NIR-Red-Green'),
        ((8,4,3), 'SWIR-Red-Green')
    ]
    
    for bands, name in false_color_combinations:
        false_color = create_rgb_image(processed_img, bands)
        plt.figure(figsize=(25, 25))
        plt.imshow(false_color)
        plt.title(f'False Color Composite ({name})')
        plt.axis('off')
        plt.savefig(os.path.join(output_directory, f'false_color_{name}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        fc_zoom1 = create_rgb_image(processed_img[2500:3000, 500:1000], bands)
        plt.figure(figsize=(25, 25))
        plt.imshow(fc_zoom1)
        plt.title(f'Zoomed False Color (Level 1) ({name})')
        plt.axis('off')
        plt.savefig(os.path.join(output_directory, f'zoomed_false_color_level1_{name}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        fc_zoom2 = create_rgb_image(processed_img[2700:3000, 700:1000], bands)
        plt.figure(figsize=(15, 15))
        plt.imshow(fc_zoom2)
        plt.title(f'Zoomed False Color (Level 2) ({name})')
        plt.axis('off')
        plt.savefig(os.path.join(output_directory, f'zoomed_false_color_level2_{name}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f'Saved {name} composition with zoomed versions')

def save_individual_bands(processed_img, output_directory, band_names):
    for i, band_name in enumerate(band_names):
        if i < processed_img.shape[2]:
            band_img = processed_img[:,:,i]
            output_path = os.path.join(output_directory, f'{band_name}.png')
            
            plt.figure(figsize=(10, 10))
            plt.imshow(band_img, cmap='gray')
            plt.axis('off')
            plt.title(band_name)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f'Saved individual band: {band_name}')

def main():
    input_directory = r'D:/MPR'  
    output_directory = 'airport_images'
    
    print("Loading and stacking bands...")
    multispectral_img, metadata, band_names = stack_bands_with_georeference(input_directory)
    
    print("\nProcessing data...")
    processed_img = process_multispectral_data(multispectral_img)
    
    print("\nCreating visualizations...")
    visualize_processed_data(processed_img, output_directory)
    
    print("\nSaving individual bands as viewable images...")
    save_individual_bands(processed_img, output_directory, band_names)
    
    output_geotiff = os.path.join(output_directory, 'processed_multispectral.tif')
    
    tiff.imwrite(os.path.join(output_directory, 'processed_multispectral_viewable.tif'), 
                processed_img, photometric='rgb')
    
    metadata.update({
        'count': processed_img.shape[2],
        'dtype': processed_img.dtype
    })
    
    with rasterio.open(output_geotiff, 'w', **metadata) as dst:
        for i in range(processed_img.shape[2]):
            dst.write(processed_img[:,:,i], i+1)
    
    print(f"\nProcessing complete! Results saved to: {os.path.abspath(output_directory)}")
    print(f"Processed GeoTIFF saved as: {output_geotiff}")
    print(f"A more viewable version is also saved as: {os.path.join(output_directory, 'processed_multispectral_viewable.tif')}")

if __name__ == "__main__":
    main()
