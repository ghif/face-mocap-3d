import sys
import os

# Ensure the user site-packages are in sys.path
user_site = os.path.expanduser('/Users/mghifary/.local/lib/python3.11/site-packages')
if user_site not in sys.path:
    sys.path.append(user_site)

import mediapipe as mp
from mediapipe import solutions
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

import time
import numpy as np
import cv2

import bpy

base_options = python.BaseOptions(
    model_asset_path="/Users/mghifary/Work/Code/AI/talkinghead-3d/models/face_landmarker_v2_with_blendshapes.task",
    delegate="CPU",
)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(options)

blendshape_names = np.load("/Users/mghifary/Work/Code/AI/talkinghead-3d/resources/blendshape_names.npy")

def draw_landmarks_on_image(rgb_image, detection_result):
    """
    Draw landmarks on the image.
    
    ArgsL
        rgb_image (numpy.ndarray): The input RGB image on which landmarks are to be drawn.
        detection_result: The detection result from the face landmark model, containing face landmarks.

        Returns:
            annotated_image (numpy.ndarray): The image with the landmarks drawn on it.
    """
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

     # Check if any face landmarks were detected
    if not face_landmarks_list:
        return annotated_image

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style())

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_iris_connections_style())

    return annotated_image

class AnimOperator(bpy.types.Operator):
    """
    OpenCV Animation Operator for Blender.

    This operator captures video frames from a camera using OpenCV, processes the frames to detect face landmarks, and updates a 3D model's blendshapes in Blender based on the detected landmarks. 
    
    The processed frames are also displayed in Blender's UV/Image Editor.

    Attributes:
        bl_idname (str): Blender operator ID name.
        bl_label (str): Blender operator label.
        _timer (bpy.types.Timer): Timer for the modal operator.
        _cap (cv2.VideoCapture): OpenCV video capture object.
        _image (bpy.types.Image): Blender image object to display the camera feed.
        width (int): Width of the camera feed and Blender image.
        height (int): Height of the camera feed and Blender image.
        stop (bpy.props.BoolProperty): Property to stop the operator.
    Methods:
        modal(context, event):
            Handles events in the modal operator, including reading frames from the camera, processing them, and updating the 3D model and Blender image.
        found_image_editor(context):
            Checks if the UV/Image Editor displaying the image is open.
        init_camera():
            Initializes the camera for capturing video frames.
        stop_playback(scene):
            Stops the playback when the animation reaches the end frame.
        execute(context):
            Executes the operator, initializing the camera and Blender image, and setting up the modal handler.
        open_image_editor(context):
            Opens a new UV/Image Editor area in Blender and assigns the image to it.
        cancel(context):
            Cancels the operator, stopping the timer, releasing the camera, and removing the Blender image if necessary.
    """

    
    bl_idname = "wm.opencv_operator"
    bl_label = "OpenCV Animation Operator"
    
    _timer = None
    _cap = None
    _image = None
    
    width = 800
    height = 600
    
    stop :bpy.props.BoolProperty()
    
    def modal(self, context, event):
        if (event.type in {'RIGHTMOUSE', 'ESC'}) or self.stop == True:
            self.cancel(context)
            return {'CANCELLED'}
        
        if event.type == 'TIMER':
            ret, frame = self._cap.read()
            
            if not ret or frame is None or frame.size == 0:
                self.report({'ERROR'}, "[modal] Failed to read frame from camera.")
                self.cancel(context)
                return {'CANCELLED'}
            
            # Show camera image in a window
            frame_out = cv2.resize(frame, (self.width, self.height))
            
            # Convert frame to the required format
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=frame_out
            )
            
            # Face landmark tracking
            detection_results = detector.detect(mp_image)
            frame_out = draw_landmarks_on_image(frame_out, detection_results)
            
            # Predict Blendshapes
            o = bpy.context.active_object # reference the active object
            
            blendshape_weights = np.zeros(len(blendshape_names))
            if len(detection_results.face_blendshapes) > 0:
                blendshape_weights = np.array([blendshape.score for blendshape in detection_results.face_blendshapes[0]])
                
            # Assign Blendshapes to the face 3D model
            named_weights = dict(zip(blendshape_names[1:], blendshape_weights[1:]))
            for k, v in named_weights.items():
                o.data.shape_keys.key_blocks[k].value = v
            
            frame_out = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGBA)
            
            # flip vertically to match Blender's coordinate system
            frame_out = cv2.flip(frame_out, 0)
            
            # flatten the array and normalize pixel values
            frame_flat = (frame_out.flatten() / 255.0).astype(np.float32)
            
            # update Blender image
            self._image.pixels = frame_flat.tolist()
            self._image.update()
                
            if not self.found_image_editor(context):
                # The Image Editor displaying the image is closed
                # Remove the image if it has  no other users
                if self._image.users == 0:
                    bpy.data.images.remove(self._image)
                    self._image = None
            
        return {'PASS_THROUGH'}
    
    def found_image_editor(self, context):
        found_image_editor = False
        for area in context.screen.areas:
            if area.type == 'IMAGE_EDITOR':
                space = area.spaces.active
                if space.image == self._image:
                    found_image_editor = True
                    break
                
        return found_image_editor
    
    def init_camera(self):
        self._cap = cv2.VideoCapture(0)
        
        if not self._cap.isOpened():
            self.report({'ERROR'}, "[init_camera] Failed to open camera.")
            return {'CANCELLED'}
        else:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            time.sleep(1.0)
    
    def stop_playback(self, scene):
        print(format(scene.frame_current) + " / " + format(scene.frame_end))
        if scene.frame_current == scene.frame_end:
            bpy.ops.screen.animation_cancel(restore_frame=False)
        
    def execute(self, context):
        # Initialize camera
        self.init_camera()
        
        # Initialize a new image in Blender
        self._image = bpy.data.images.new("Camera Feed", width=self.width, height=self.height, alpha=True)
        
        # Assign the image to the UV/Image Editor
        found_image_editor = self.found_image_editor(context)
        
        if not found_image_editor:
            # Open a new UV/Image Editor area
            self.open_image_editor(context)
        
        bpy.app.handlers.frame_change_pre.append(self.stop_playback)

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.01, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}
    
    def open_image_editor(self, context):
        # Split the current area and create a new UV/Image Editor area
        screen = context.screen
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                override = {'window': context.window, 'screen': screen, 'area': area}
                with context.temp_override(**override):
                    bpy.ops.screen.area_split(direction='VERTICAL', factor=0.5)
                
                # After splitting, Blender adds the new area to the areas list
                new_area = screen.areas[-1]
                new_area.type = 'IMAGE_EDITOR'
                new_area.spaces.active.image = self._image
                
                break
                

    def cancel(self, context):
        if self._timer:
            wm = context.window_manager
            wm.event_timer_remove(self._timer)
            self._timer = None
            
        if self._cap:
            self._cap.release()
            self._cap = None
        
        if self._image:
            if self._image.users == 0:
                bpy.data.images.remove(self._image)
            
            self._image = None

def register():
    try:
        unregister()
    except Exception:
        pass
    
    bpy.utils.register_class(AnimOperator)
    
def unregister():
    bpy.utils.unregister_class(AnimOperator)

if __name__ == "__main__":
    register()
    
    # test call
#    bpy.ops.wm.opencv_operator()