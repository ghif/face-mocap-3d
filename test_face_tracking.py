import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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

def draw_blendshape_weights(blendshape_names, blendshape_weights, target_shape=(540, 960)):
    fig, ax1 = plt.subplots(1, 1, figsize=(28, 12))
    ax1.barh(blendshape_names, blendshape_weights)
    ax1.set_xlim(0, 1)
    ax1.set_ylabel("Blendshapes", fontsize=20)
    ax1.set_xlabel("Weights", fontsize=20)
    ax1.tick_params(axis='y', labelsize=15)
    ax1.tick_params(axis='x', labelsize=15)

    # Convert the figure to an image
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)

    # Reshape
    width, height = fig.canvas.get_width_height()
    # image2 = image2.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # img = img.reshape((height * 2, width * 2, 4))
    img = img.reshape((height, width, 4))

    # Resize image2 to match image1
    img = cv2.resize(img, (target_shape[1], target_shape[0]))
    img = img[:, :, :3]
    return img

# Define MediaPipe face landmark detector
base_options = python.BaseOptions(
    model_asset_path="resources/face_landmarker_v2_with_blendshapes.task",
    delegate="CPU",
)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# Load blendshape names
blendshape_names = np.load("resources/blendshape_names.npy")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Failed to open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame.")
        break

    # Convert frame to the required format
    mp_image = mp.Image(
       image_format=mp.ImageFormat.SRGB,
       data=frame
    )
    
    # Detect the face landmarks and predict the blendshape weights
    detection_result = detector.detect(mp_image)

    # Draw landmarks on the image    
    annotated_image = draw_landmarks_on_image(frame, detection_result)

    # Draw blendshape weights
    blendshape_weights = np.zeros(len(blendshape_names))

    if len(detection_result.face_blendshapes) == 0:
        continue

    blendshape_weights = np.array([blendshape.score for blendshape in detection_result.face_blendshapes[0]])
    blendshape_graph = draw_blendshape_weights(
        blendshape_names, 
        blendshape_weights,
        target_shape=(annotated_image.shape[0], annotated_image.shape[1])
    )


    final_image = np.vstack([annotated_image, blendshape_graph])

    # Display the output
    cv2.imshow('Camera Feed', final_image)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()