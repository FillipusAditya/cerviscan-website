"""
============================================
📌 Single Image Feature Extraction (Python)
============================================

Description:
------------
This program extracts multiple color moment and texture features from a single 
segmented image. It is designed for website or application use cases where 
features from individual images are required for tasks such as image 
classification, clustering, or retrieval.

The extracted features include color moments from different color spaces (RGB, YUV, LAB) 
and several texture features (GLRLM, TAMURA, LBP). The program returns the 
results as a Pandas DataFrame for easy integration into data analysis or machine 
learning pipelines.

Features:
---------
✅ Supports extraction of color moments from RGB, YUV, and LAB color spaces  
✅ Supports multiple texture descriptors: LBP, GLRLM, TAMURA
✅ Flexible selection of feature types (color spaces and texture methods)  
✅ Outputs results as a clean Pandas DataFrame  
✅ Automatically removes constant-valued columns  

Usage:
------
1. Import the `SingleImageFeatureExtractor` class.
2. Initialize the extractor: `extractor = SingleImageFeatureExtractor()`.
3. Call `extractor.extract_features(image_path)` with the path to the input image.
4. (Optional) Specify desired color spaces or texture feature sets.

Example:
--------
segmented_path = "path/to/segmented_image.png"  
extractor = SingleImageFeatureExtractor()  
image_features = extractor.extract_features(segmented_path)  
print(image_features)

Author: Fillipus Aditya Nugroho
============================================
"""

import pandas as pd

# Import all feature extraction functions & name getters
from model.rgb_color_moment import get_rgb_color_moment_features, get_rgb_color_moment_feature_names
from model.yuv_color_moment import get_yuv_color_moment_features, get_yuv_color_moment_feature_names
from model.lab_color_moment import get_lab_color_moment_features, get_lab_color_moment_feature_names
from model.glrlm_feature_extraction import get_glrlm_features, get_glrlm_feature_names
from model.tamura_feature_extraction import get_tamura_features, get_tamura_feature_names
from model.lbp_feature_extraction import get_lbp_features, get_lbp_feature_names


class SingleImageFeatureExtractor:
    """
    Extract multiple color moment and texture features
    for a single input image (website use case).
    """

    def __init__(self):
        # Mapping for color moment features
        self.color_moment_features = {
            'RGB': (get_rgb_color_moment_features, get_rgb_color_moment_feature_names),
            'YUV': (get_yuv_color_moment_features, get_yuv_color_moment_feature_names),
            'LAB': (get_lab_color_moment_features, get_lab_color_moment_feature_names),
        }

        # Mapping for texture features
        self.texture_features = {
            'GLRLM': (get_glrlm_features, get_glrlm_feature_names),
            'TAMURA': (get_tamura_features, get_tamura_feature_names),
            'LBP': (get_lbp_features, get_lbp_feature_names),
        }

    def extract_features(self, image_path, color_spaces=None, texture_features=None):
        """
        Extract specified features for a single image.

        Parameters
        ----------
        image_path : str
            Path to the segmented input image.
        color_spaces : list of str, optional
            Options: ['RGB', 'YUV', 'LAB']. Default = all.
        texture_features : list of str, optional
            Options: ['LBP', 'GLRLM', 'TAMURA']. 
            Default = ['LBP', 'GLRLM', 'TAMURA'].

        Returns
        -------
        df_features : pd.DataFrame
            Extracted features in a DataFrame (1 row).
        """
        if color_spaces is None:
            color_spaces = ['RGB', 'YUV', 'LAB']
        if texture_features is None:
            texture_features = ['LBP', 'GLRLM', 'TAMURA']

        features = []
        features_name = []

        # Extract color moment features
        for key in color_spaces:
            if key in self.color_moment_features:
                get_features, get_names = self.color_moment_features[key]
                features.extend(get_features(image_path))
                features_name.extend(get_names())

        # Extract texture features
        for key in texture_features:
            if key in self.texture_features:
                get_features, get_names = self.texture_features[key]
                features.extend(get_features(image_path))
                features_name.extend(get_names())

        # Convert to DataFrame
        df_features = pd.DataFrame([features], columns=features_name)
        df_features = df_features.loc[:, (df_features != 1).any()]  # drop constant cols

        return df_features