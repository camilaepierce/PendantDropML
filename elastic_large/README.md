This repo contains only the file difference from Tensiomet2 needed to generate `N` synthetic drops.

<b>Use as follows:</b><br>
Download and follow ReadMe for the Tensiomet2 (https://github.com/NickJaensson/tensiomet2) MATLAB library.

Replace the following files with the verions within this repository:
`example_simple.m`
`parameters_simple.m`

To the main directory add `create_samples.`
To the `/src` directory add `plot_shape_full.m`

Add folder `/sample_generation`, and within that add `/test_images`, `/test_data_rz`, and `/test_data_params`.
*Folders can be renamed as long as the new names/structure is reflected in the file-saving segment of `create_samples`.

Add all folders and new file to MATLAB path by selecting, right-clicking, and choosing "Add to path".

Change the `N` variable within `create_samples` to produce `N` synthetic drops. The images, parameters, and r-z coordinates will be saved to the `/sample_generation` folder.

If generating more than 999 samples, update the number of siginificant digits within the filenames of the file-saving segment of `create_samples`.
