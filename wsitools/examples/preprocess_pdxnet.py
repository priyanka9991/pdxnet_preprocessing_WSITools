from wsitools.tissue_detection.tissue_detector import TissueDetector
from wsitools.patch_extraction.patch_extractor import ExtractorParameters, PatchExtractor
import os
import matplotlib.pyplot as plt
from PIL import Image
import staintools
from PIL import Image
import numpy as np
import logging
import sys

# setup logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger('WSI Preprocessing')

def patch_extractor(wsi_fn, patch_dir, log_dir, extractor_params):
    logger.info(str('Extracting patches for ' + wsi_fn) )
    if not os.path.exists(patch_dir):
        os.makedirs(patch_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    print(log_dir)
    print(patch_dir)
    # Define the parameters for Patch Extraction, including generating an thumbnail from which to traverse over to find
    # tissue.
    parameters = ExtractorParameters(patch_dir, # Where the patches should be extracted to
        save_format = extractor_params['save_format'],                      # Can be '.jpg', '.png', or '.tfrecord'
        sample_cnt = extractor_params['sample_cnt'],                           # Limit the number of patches to extract (-1 == all patches)
        patch_size = extractor_params['patch_size'],                          # Size of patches to extract (Height & Width)
        rescale_rate = extractor_params['rescale_rate'],                        # Fold size to scale the thumbnail to (for faster processing)
        patch_filter_by_area = extractor_params['patch_filter_by_area'],                # Amount of tissue that should be present in a patch
        with_anno = extractor_params['with_anno'],                          # If true, you need to supply an additional XML file
        extract_layer = extractor_params['extract_layer'],                          # OpenSlide Level
        log_dir=log_dir
        )

    # Choose a method for detecting tissue in thumbnail image
    tissue_detector = TissueDetector("LAB_Threshold",   # Can be LAB_Threshold or GNB
        threshold = extractor_params['threshold'],                                   # Number from 1-255, anything less than this number means there is tissue
        training_files = extractor_params['training_files']                             # Training file for GNB-based detection
        )

    # Create the extractor object
    patch_extractor = PatchExtractor(tissue_detector,
        parameters,
        feature_map = extractor_params['feature_map'],                       # See note below
        annotations = extractor_params['annotations']                        # Object of Annotation Class (see other note below)
        )
    patch_extractor.extract(wsi_fn)


def normalize_patches(template_img, slide_name, normalized_patches_dir, patch_dir):
    logger.info(str('Normalizing patches for slide: '+ slide_name))
    if not os.path.exists(normalized_patches_dir):
        os.makedirs(normalized_patches_dir)
    img_fn_list = os.listdir(str(patch_dir+'/'+slide_name))
    skipped_patches = []
    for img_fn in img_fn_list:
        try:
            target = staintools.read_image(template_img)
            to_transform = staintools.read_image(str(patch_dir+'/'+slide_name+'/'+img_fn))

            # Standardize brightness (optional, can improve the tissue mask calculation)
            target = staintools.LuminosityStandardizer.standardize(target)
            to_transform = staintools.LuminosityStandardizer.standardize(to_transform)

            # Stain normalize
            normalizer = staintools.StainNormalizer(method='vahadane')      
            normalizer.fit(target)
            transformed = normalizer.transform(to_transform)
        except staintools.miscellaneous.exceptions.TissueMaskException: 
            #Sometimes, the artifacts will be recognized as tissue (as we down sampled the WSI to detect the tissue region), 
            # but when we extract the tiles (which is at high resolution), there could actually be no tissue at all.
            logger.warning(f"TissueMaskException encountered for image {img_fn}, skipping...")
            skipped_patches.append(img_fn)
            continue

        # Convert numpy array to PIL Image and save
        transformed_image = Image.fromarray(transformed.astype('uint8'), 'RGB')
        if not os.path.exists(os.path.join(normalized_patches_dir, slide_name)):
            os.makedirs(os.path.join(normalized_patches_dir, slide_name))
        sv_fn = os.path.join(normalized_patches_dir, slide_name, img_fn)
        transformed_image.save(sv_fn)
    # Save skipped patches list to file
    if skipped_patches:
        skipped_patches_file = os.path.join(normalized_patches_dir, slide_name, 'skipped_patches.txt')
        with open(skipped_patches_file, 'w') as f:
            f.write('\n'.join(skipped_patches))
        logger.info(f"Saved list of {len(skipped_patches)} skipped patches to {skipped_patches_file}")

def get_svs_files(pdx_dir):
    svs_files = []
    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(pdx_dir):
        for file in files:
            if file.endswith('.svs') or file.endswith('.tif'):
                # Get the full path
                full_path = os.path.join(root, file)
                # Get the relative path after pdx_dir
                relative_path = os.path.relpath(full_path, pdx_dir)
                svs_files.append(relative_path)
    return svs_files
    
def run():
    pdx_dir = '/nfs/ml_lab/projects/Pilot1_PreclinicalHPC/digPath_Data/PDXNet'
    preprocessed_outdir = '/nfs/ml_lab/projects/Pilot1_PreclinicalHPC/priyanka/PDX_net_Preprocessed'
    extractor_params = {
            'save_format':'.png',                      # Can be '.jpg', '.png', or '.tfrecord'
            'sample_cnt': -1,                           # Limit the number of patches to extract (-1 == all patches)
            'patch_size': 128,                          # Size of patches to extract (Height & Width)
            'rescale_rate' : 128,                        # Fold size to scale the thumbnail to (for faster processing)
            'patch_filter_by_area' : 0.5,                # Amount of tissue that should be present in a patch
            'with_anno' : True,                          # If true, you need to supply an additional XML file
            'extract_layer' : 0,                          # OpenSlide Level
            'threshold' : 85,                            # Number from 1-255, anything less than this number means there is tissue
            'training_files' : None,                       # Training file for GNB-based detection
            'feature_map' : None,                       
            'annotations' : None  
    }
    studies = os.listdir(pdx_dir) # ['WISTAR', 'WUSTL', 'BCM', 'JAX', 'HCI', 'MDA', 'PDMR'] - 7 studies
    study = studies[0] # CHANGE THIS # or make a loop for all the studies
    study_dir = os.path.join(pdx_dir, study)
    preprocessed_outdir = os.path.join(preprocessed_outdir, study) # Output dir to save preprocessed images
    patch_extract = True # Do you want to extract patches?
    normalize = False # Do you want to normalize the patches?

    #Patch extraction and normalization
    svs_files = get_svs_files(study_dir) # Get all svs images in all directories and subdirectories
    all_patch_dirs=[]
    template_img = './template_pdx_net.png' 
    # Reference image - You can randomly pick any refence image which contains both tumor and stroma
    # https://github.com/Peter554/StainTools/blob/master/staintools/preprocessing/luminosity_standardizer.py#L10
    for file in svs_files:
        patch_dir=os.path.join(preprocessed_outdir, os.path.dirname(file), 'patches')
        if patch_extract:
            wsi_fn=os.path.join(study_dir, file)
            all_patch_dirs.append(patch_dir)
            log_dir=os.path.join(preprocessed_outdir, os.path.dirname(file), 'logs')
            os.makedirs(patch_dir, exist_ok=True)
            os.makedirs(log_dir, exist_ok=True)
            # Patch extraction
            patch_extractor(wsi_fn, patch_dir, log_dir, extractor_params) 
        if normalize:
        # Patch normalization
            normalized_patches_dir = os.path.join(preprocessed_outdir, os.path.dirname(file), 'normalized_patches')
            slide_name=file.split('/')[-1][:-4]
            normalize_patches(template_img, slide_name, normalized_patches_dir, patch_dir)

if __name__ == "__main__":
    run()

