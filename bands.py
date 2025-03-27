#This code gives us the output of the processed false color composite images of the region of interest 
#We can zoom on our AOI and change it accordingly

import numpy as np
import tifffile as tiff
import os
from sklearn.preprocessing import MinMaxScaler
from skimage import exposure
import matplotlib.pyplot as plt
import rasterio
import spectral.io.envi as envi
from spectral import imshow
import warnings
warnings.filterwarnings('ignore')

def stack_bands_with_georeference(directory):
    """
    Stack multiple band images while preserving geospatial information
    """
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
    """
    Process the multispectral image using the specified normalization approach
    """
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

def visualize_processed_data(processed_img, output_directory):
    """
    Create visualizations using spectral library
    """
    os.makedirs(output_directory, exist_ok=True)
    
    plt.figure(figsize=(25, 25))
    view = imshow(processed_img, (3,2,1))  
    plt.title('RGB Composite (Red-Green-Blue)')
    plt.savefig(os.path.join(output_directory, 'rgb_composite_spectral.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    false_color_combinations = [
        ((7,3,2), 'NIR-Green-Blue'),
        ((7,4,3), 'NIR-Red-Green'),
        ((8,4,3), 'SWIR-Red-Green')
    ]
    
    for bands, name in false_color_combinations:
        plt.figure(figsize=(25, 25))
        view = imshow(processed_img, bands)
        plt.title(f'False Color Composite ({name})')
        plt.savefig(os.path.join(output_directory, f'false_color_{name}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f'Saved {name} composition')

def main():
    input_directory = r'D:/MPR'  #change according to your requirements
    output_directory = 'processed_bands_geospatial'
    
    print("Loading and stacking bands...")
    multispectral_img, metadata, band_names = stack_bands_with_georeference(input_directory)
    
    print("\nProcessing data...")
    processed_img = process_multispectral_data(multispectral_img)
    
    print("\nCreating visualizations...")
    visualize_processed_data(processed_img, output_directory)
    
    output_geotiff = os.path.join(output_directory, 'processed_multispectral.tif')
    metadata.update({
        'count': processed_img.shape[2],
        'dtype': processed_img.dtype
    })
    
    with rasterio.open(output_geotiff, 'w', **metadata) as dst:
        for i in range(processed_img.shape[2]):
            dst.write(processed_img[:,:,i], i+1)
    
    print(f"\nProcessing complete! Results saved to: {os.path.abspath(output_directory)}")
    print(f"Processed GeoTIFF saved as: {output_geotiff}")

if __name__ == "__main__":
    main()
