# Real-time 3D Face Motion Capture with Blender
This repository provides a simple solution for real-time 3D face motion capture using Blender. It leverages MediaPipe for facial motion detection using face landmarks and OpenCV for real-time video processing. The captured motion is mapped to ARKit-compatible blendshapes, enabling expressive and dynamic facial animations for 3D models.

## Requirements
- Blender v4.2.3
- MediaPipe v0.10.18
- OpenCV-python v4.10.0

## Running the Motion Capture


### 1. Prepare a 3D face blendshape 

An example of 3D face blendshape is the [MetaHuman Head from Epic Games](https://bazaar.blendernation.com/listing/metahuman-head-52-blendshapes-arkit-ready/) that can be downloaded for free. 
That blendshape is compatible with [ARKit](https://arkit-face-blendshapes.com/) configuration that comprises of [52 face shapes](https://github.com/elijah-atkins/ARKitBlendshapeHelper).

### 2. Setup Blender python scripts
- __Animation operator__: Create a script named anim_operator.py in Blender. Copy and adjust the code from the anim_operator.py file provided in this repository to animate blendshapes based on real-time input.
- __View panel__: Create a script named anim_view.py in Blender. Copy and adjust the code from the anim_view.py file provided in this repository. This script adds a custom panel to the Blender interface, allowing you to control animations directly from the 3D Viewport.

### 3. Start real-time motion capture
- Load the 3D model into Blender
- Ensure the required scripts (anim_operator.py, anim_view.py) are registered.
- Use a webcam to capture facial movements, which are then animated in real-time.

## Troubleshooting
- __Webcam Issues__: Ensure your webcam is properly connected and recognized by the system.
- __Library Compatibility__: Check that you have installed the correct versions of the required Python libraries.
- __Model Configuration__: Verify that your 3D model uses ARKit-compatible blendshapes.

## Contributions
Contributions are welcome! Whether it's improving the scripts, adding new features, or reporting bugs, feel free to fork this repository and submit a pull request.