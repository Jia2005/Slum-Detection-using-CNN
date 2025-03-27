#This is an extension to airport.py file
#In this, we also have slum detection done and it is saved inside the output directory in a folder with both levels of zoom (with smoothing done)

import numpy as np
import tifffile as tiff
import os
from sklearn.preprocessing import MinMaxScaler
from skimage import exposure, feature
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import rasterio
import warnings
warnings.filterwarnings('ignore')
from sklearn.cluster import KMeans
from scipy import ndimage
from tqdm import tqdm
import time

def print_progress(iteration, total, prefix='Progress:', suffix='Complete', decimals=1, length=50, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total:
        print()

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
            print_progress(idx + 1, len(band_names), prefix='Loading Bands:', suffix=f'Loaded {band_name}')
        except Exception as e:
            print(f'Error loading {band_name}: {str(e)}')
    
    return multispectral_img, metadata, band_names

def process_multispectral_data(multispectral_img):
    print_progress(1, 4, prefix='Processing Data:', suffix='Flattening image')
    orig_shape = multispectral_img.shape
    flat_img = np.reshape(multispectral_img, (orig_shape[0] * orig_shape[1], orig_shape[2]))
    
    print_progress(2, 4, prefix='Processing Data:', suffix='Normalizing data')
    scaler = MinMaxScaler()
    norm_flat = scaler.fit_transform(flat_img)
    
    print_progress(3, 4, prefix='Processing Data:', suffix='Reshaping data')
    norm_img = np.reshape(norm_flat, orig_shape)
    
    print_progress(4, 4, prefix='Processing Data:', suffix='Equalizing histogram')
    hist_eq = exposure.equalize_hist(norm_img)
    hist_eq = (hist_eq * 255).astype('uint8')
    
    return hist_eq

def create_rgb_image(img, bands=(3,2,1)):
    rgb = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for i in range(3):
        if bands[i] < img.shape[2]:
            rgb[:,:,i] = img[:,:,bands[i]]
    return rgb

def calculate_features(image):
    features = []
    
    print_progress(1, 9, prefix='Feature Calculation:', suffix='Calculating NDVI')
    nir, red = image[:,:,7], image[:,:,2]
    ndvi = (nir - red) / (nir + red + 1e-8)
    features.append(ndvi)
    
    print_progress(2, 9, prefix='Feature Calculation:', suffix='Calculating NDBI')
    swir = image[:,:,8]
    ndbi = (swir - nir) / (swir + nir + 1e-8)
    features.append(ndbi)
    
    window_size = 16
    step_size = 8
    
    band_indices = [2, 3, 7]
    for i, band in enumerate(band_indices):
        base_progress = 3 + i * 2
        
        print_progress(base_progress, 9, prefix='Feature Calculation:', suffix=f'Processing band {band} texture')
        band_small = image[::step_size, ::step_size, band]
        band_normalized = exposure.rescale_intensity(band_small, out_range=(0, 255))
        band_uint8 = band_normalized.astype(np.uint8)
        
        glcm = feature.graycomatrix(band_uint8, [1], [0], 256)
        contrast = feature.graycoprops(glcm, 'contrast')[0, 0]
        features.append(np.full_like(ndvi, contrast))
        
        print_progress(base_progress + 1, 9, prefix='Feature Calculation:', suffix=f'Processing band {band} variance')
        variance = ndimage.generic_filter(image[:,:,band], np.var, size=window_size)
        features.append(variance)
    
    print_progress(9, 9, prefix='Feature Calculation:', suffix='Features complete')
    return np.stack(features, axis=-1)

def detect_slums_kmeans(processed_img, region=None, n_clusters=3):
    if region:
        img_section = processed_img[region[0]:region[1], region[2]:region[3], :]
    else:
        img_section = processed_img
    
    print_progress(1, 3, prefix='KMeans Clustering:', suffix='Preparing data')
    orig_shape = img_section.shape
    flat_img = np.reshape(img_section, (orig_shape[0] * orig_shape[1], orig_shape[2]))
    
    print_progress(2, 3, prefix='KMeans Clustering:', suffix='Running clustering')
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(flat_img)
    labels = kmeans.labels_
    
    print_progress(3, 3, prefix='KMeans Clustering:', suffix='Reshaping results')
    clustered = np.reshape(labels, (orig_shape[0], orig_shape[1]))
    
    return clustered, kmeans

def smooth_slum_map(slum_map, filter_size=5):
    print_progress(1, 1, prefix='Smoothing:', suffix=f'Applying median filter (size={filter_size})')
    smoothed = ndimage.median_filter(slum_map, size=filter_size)
    return smoothed

def visualize_processed_data(processed_img, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    
    total_steps = 9
    current_step = 0
    
    current_step += 1
    print_progress(current_step, total_steps, prefix='Visualization:', suffix='Creating RGB composite')
    rgb_image = create_rgb_image(processed_img, (3,2,1))
    plt.figure(figsize=(25, 25))
    plt.imshow(rgb_image)
    plt.title('RGB Composite (Red-Green-Blue)')
    plt.axis('off')
    plt.savefig(os.path.join(output_directory, 'rgb_composite.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    current_step += 1
    print_progress(current_step, total_steps, prefix='Visualization:', suffix='Creating zoomed RGB level 1')
    zoomed_rgb1 = create_rgb_image(processed_img[2500:3000, 500:1000], (3,2,1))
    plt.figure(figsize=(25, 25))
    plt.imshow(zoomed_rgb1)
    plt.title('Zoomed RGB Composite (First Level)')
    plt.axis('off')
    plt.savefig(os.path.join(output_directory, 'zoomed_rgb_level1.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    
    current_step += 1
    print_progress(current_step, total_steps, prefix='Visualization:', suffix='Creating zoomed RGB level 2')
    zoomed_rgb2 = create_rgb_image(processed_img[2700:3000, 700:1000], (3,2,1))
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
    
    for i, (bands, name) in enumerate(false_color_combinations):
        current_step += 1
        print_progress(current_step, total_steps, prefix='Visualization:', suffix=f'Creating {name} composition')
        
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

    current_step = total_steps
    print_progress(current_step, total_steps, prefix='Visualization:', suffix='Complete')

def save_individual_bands(processed_img, output_directory, band_names):
    os.makedirs(output_directory, exist_ok=True)
    
    for i, band_name in enumerate(band_names):
        if i < processed_img.shape[2]:
            print_progress(i + 1, len(band_names), prefix='Saving Bands:', suffix=f'Processing {band_name}')
            
            band_img = processed_img[:,:,i]
            output_path = os.path.join(output_directory, f'{band_name}.png')
            
            plt.figure(figsize=(10, 10))
            plt.imshow(band_img, cmap='gray')
            plt.axis('off')
            plt.title(band_name)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

def visualize_slum_detection(processed_img, slum_areas, smoothed_slums, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    
    total_steps = 10
    current_step = 0
    
    current_step += 1
    print_progress(current_step, total_steps, prefix='Slum Detection Viz:', suffix='Creating RGB base')
    rgb_image = create_rgb_image(processed_img, (3,2,1))
    
    plt.figure(figsize=(20, 20))
    plt.imshow(rgb_image)
    plt.title('RGB Composite Image')
    plt.axis('off')
    plt.savefig(os.path.join(output_directory, 'rgb_base.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    current_step += 1
    print_progress(current_step, total_steps, prefix='Slum Detection Viz:', suffix='Creating raw slum map')
    plt.figure(figsize=(20, 20))
    plt.imshow(slum_areas, cmap='viridis')
    plt.title('Slum Areas - KMeans Clustering')
    plt.axis('off')
    plt.colorbar()
    plt.savefig(os.path.join(output_directory, 'slum_detection_raw.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    
    current_step += 1
    print_progress(current_step, total_steps, prefix='Slum Detection Viz:', suffix='Creating smoothed slum map')
    plt.figure(figsize=(20, 20))
    plt.imshow(smoothed_slums, cmap='viridis')
    plt.title('Slum Areas - Smoothed')
    plt.axis('off')
    plt.colorbar()
    plt.savefig(os.path.join(output_directory, 'slum_detection_smoothed.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    
    current_step += 1
    print_progress(current_step, total_steps, prefix='Slum Detection Viz:', suffix='Creating slum overlay')
    plt.figure(figsize=(20, 20))
    plt.imshow(rgb_image)
    mask = np.ma.masked_where(smoothed_slums != np.max(smoothed_slums), smoothed_slums)
    plt.imshow(mask, cmap='autumn', alpha=0.6)
    plt.title('Slum Areas Overlay - Identified Slums')
    plt.axis('off')
    plt.savefig(os.path.join(output_directory, 'slum_detection_overlay.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    
    regions_of_interest = [
        ((2300, 2800, 1200, 1700), 'Area1'),
        ((2500, 3000, 500, 1000), 'Area2'),
        ((2700, 3000, 700, 1000), 'Area3')
    ]
    
    for roi_idx, (roi, name) in enumerate(regions_of_interest):
        y1, y2, x1, x2 = roi
        if (y1 < processed_img.shape[0] and y2 <= processed_img.shape[0] and 
            x1 < processed_img.shape[1] and x2 <= processed_img.shape[1]):
            
            current_step += 1
            print_progress(current_step, total_steps, prefix='Slum Detection Viz:', suffix=f'Processing ROI {name} - RGB')
            
            roi_rgb = create_rgb_image(processed_img[y1:y2, x1:x2], (3,2,1))
            roi_slums = smoothed_slums[y1:y2, x1:x2]
            
            plt.figure(figsize=(15, 15))
            plt.imshow(roi_rgb)
            plt.title(f'ROI {name} - RGB Image')
            plt.axis('off')
            plt.savefig(os.path.join(output_directory, f'roi_{name}_rgb.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            current_step += 1
            print_progress(current_step, total_steps, prefix='Slum Detection Viz:', suffix=f'Processing ROI {name} - Slums')
            
            plt.figure(figsize=(15, 15))
            plt.imshow(roi_rgb)
            mask = np.ma.masked_where(roi_slums != np.max(roi_slums), roi_slums)
            plt.imshow(mask, cmap='autumn', alpha=0.6)
            plt.title(f'ROI {name} - Slum Detection Overlay')
            plt.axis('off')
            plt.savefig(os.path.join(output_directory, f'roi_{name}_slums.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            roi_kmeans, _ = detect_slums_kmeans(processed_img, region=roi)
            roi_smoothed = smooth_slum_map(roi_kmeans)
            
            plt.figure(figsize=(15, 15))
            plt.imshow(roi_smoothed, cmap='viridis')
            plt.title(f'ROI {name} - Dedicated Slum Analysis')
            plt.axis('off')
            plt.colorbar()
            plt.savefig(os.path.join(output_directory, f'roi_{name}_dedicated.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()

    print_progress(total_steps, total_steps, prefix='Slum Detection Viz:', suffix='Complete')

def main():
    start_time = time.time()
    input_directory = r'D:/MPR'  
    output_directory = 'satellite_analysis'
    slum_detection_directory = os.path.join(output_directory, 'slum_detection')
    
    print("==== Satellite Image Processing & Slum Detection ====")
    print(f"Starting processing at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Input directory: {input_directory}")
    print(f"Output directory: {output_directory}")
    print("====================================================")
    
    total_steps = 10
    current_step = 0
    
    current_step += 1
    print_progress(current_step, total_steps, prefix='Main Process:', suffix='Loading and stacking bands')
    multispectral_img, metadata, band_names = stack_bands_with_georeference(input_directory)
    
    current_step += 1
    print_progress(current_step, total_steps, prefix='Main Process:', suffix='Processing multispectral data')
    processed_img = process_multispectral_data(multispectral_img)
    
    current_step += 1
    print_progress(current_step, total_steps, prefix='Main Process:', suffix='Creating visualizations')
    visualize_processed_data(processed_img, output_directory)
    
    current_step += 1
    print_progress(current_step, total_steps, prefix='Main Process:', suffix='Saving individual bands')
    save_individual_bands(processed_img, output_directory, band_names)
    
    current_step += 1
    print_progress(current_step, total_steps, prefix='Main Process:', suffix='Running KMeans for slum detection')
    slums_full, _ = detect_slums_kmeans(processed_img)
    
    current_step += 1
    print_progress(current_step, total_steps, prefix='Main Process:', suffix='Smoothing slum detection')
    smoothed_slums = smooth_slum_map(slums_full, filter_size=7)
    
    current_step += 1
    print_progress(current_step, total_steps, prefix='Main Process:', suffix='Generating slum visualizations')
    visualize_slum_detection(processed_img, slums_full, smoothed_slums, slum_detection_directory)
    
    current_step += 1
    print_progress(current_step, total_steps, prefix='Main Process:', suffix='Analyzing region of interest')
    region_of_interest = (2300, 2800, 1200, 1700)
    y1, y2, x1, x2 = region_of_interest
    
    roi_img = processed_img[y1:y2, x1:x2, :]
    roi_clusters, _ = detect_slums_kmeans(roi_img, n_clusters=3)
    roi_smoothed = smooth_slum_map(roi_clusters)
    
    plt.figure(figsize=(15, 15))
    plt.imshow(roi_smoothed, cmap='gray')
    plt.title('Slums Detection - Region of Interest')
    plt.axis('off')
    plt.savefig(os.path.join(slum_detection_directory, 'roi_slums_specific.png'),
               dpi=300, bbox_inches='tight')
    plt.close()
    
    current_step += 1
    print_progress(current_step, total_steps, prefix='Main Process:', suffix='Saving processed GeoTIFF')
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
    
    current_step += 1
    print_progress(current_step, total_steps, prefix='Main Process:', suffix='Saving slum detection GeoTIFF')
    slum_output_geotiff = os.path.join(slum_detection_directory, 'slum_detection.tif')
    
    with rasterio.open(slum_output_geotiff, 'w', 
                      driver='GTiff',
                      height=smoothed_slums.shape[0],
                      width=smoothed_slums.shape[1],
                      count=1,
                      dtype=smoothed_slums.dtype) as dst:
        dst.write(smoothed_slums, 1)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("====================================================")
    print(f"Processing complete! Elapsed time: {elapsed_time:.2f} seconds")
    print(f"Results saved to: {os.path.abspath(output_directory)}")
    print(f"Slum detection maps saved to: {os.path.abspath(slum_detection_directory)}")
    print("====================================================")

if __name__ == "__main__":
    main()
