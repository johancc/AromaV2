import time
import threading
import edgeiq
import argparse
import socketio
from CVClient import CVClient

sio = socketio.Client()
models = {}
global model_choice
model_choice = "face" # Defaults to face detection.
MODELS = ["object_detect", "pose_estimation", "segmentation",
          "gender", "hand", "face", "age"]

@sio.event
def connect():
    print('[INFO] Successfully connected to server.')


@sio.event
def connect_error():
    print('[INFO] Failed to connect to server.')


@sio.event
def disconnect():
    print('[INFO] Disconnected from server.')

def displayRuntimeStatistics(model):
    print("Loaded model:\n{}\n".format(model.model_id))
    print("Engine: {}".format(model.engine))
    print("Accelerator: {}\n".format(model.accelerator))
    try:
      # These only apply to some models.
      print("Labels:\n{}\n".format(model.labels))
    except:
      pass

def initializeModels():
  for m in MODELS:
    models[m] = generateModel(m)
  
def generateModel(model_choice):
  if model_choice == 'object_detect':
    return generateObjectDetectionModel()
  elif model_choice == 'pose_estimation':
    return generatePoseEstimationModel()
  elif model_choice == 'segmentation':
    return generateSegmentationModel()
  elif model_choice == 'gender':
    return generateGenderModel()
  elif model_choice == 'age':
    return generateAgeModel() 
  elif model_choice == 'hand':
    return generateHandDetectionModel()
  elif model_choice == 'face':
    return generateFaceDetectionModel()

def generateObjectDetectionModel():
  object_detect = edgeiq.ObjectDetection("alwaysai/mobilenet_ssd")
  object_detect.load(engine=edgeiq.Engine.DNN)
  return object_detect

def generateHandDetectionModel():
  hand_detect = edgeiq.ObjectDetection("alwaysai/hand_detection")
  hand_detect.load(engine=edgeiq.Engine.DNN)
  return hand_detect
def generateFaceDetectionModel():
  face_detect = edgeiq.ObjectDetection("alwaysai/res10_300x300_ssd_iter_140000")
  face_detect.load(engine=edgeiq.Engine.DNN)
  return face_detect

def generateGenderModel():
  classification = edgeiq.Classification("alwaysai/gendernet")
  classification.load(engine=edgeiq.Engine.DNN)
  return classification

def generateAgeModel():
  age_model = edgeiq.Classification("alwaysai/agenet")
  age_model.load(engine=edgeiq.Engine.DNN)
  return age_model


def generatePoseEstimationModel():
  pose_estimation = edgeiq.PoseEstimation("alwaysai/human-pose")
  pose_estimation.load(engine=edgeiq.Engine.DNN)
  return pose_estimation

def generateSegmentationModel():
  segmentation = edgeiq.SemanticSegmentation("alwaysai/enet")
  segmentation.load(engine=edgeiq.Engine.DNN)
  return segmentation

def runModel(model, frame, model_choice):
  if model_choice == 'object_detect':
    return runObjectDetectionModel(model, frame)
  elif model_choice == 'hand':
    return runHandDetectionModel(model, frame)
  elif model_choice == 'face':
    return runFaceDetectionModel(model, frame)
  elif model_choice == 'pose_estimation':
    return runPoseEstimationModel(model, frame)
  elif model_choice == 'segmentation':
    return runSegmentationModel(model, frame)
  elif model_choice == 'gender':
    return runGenderModel(model, frame)
  elif model_choice == 'age':
    return runAgeModel(model, frame)

def _runDetectionModel(model, frame):
  results = model.detect_objects(frame, confidence_level=.5)
  frame = edgeiq.markup_image(
          frame, results.predictions, colors=model.colors)

  # Generate text to display on streamer
  text = ["Model: {}".format(model.model_id)]
  text.append(
          "Inference time: {:1.3f} s".format(results.duration))
  text.append("Objects:")

  for prediction in results.predictions:
      text.append("{}: {:2.2f}%".format(
          prediction.label, prediction.confidence * 100))
  return frame, text

def runObjectDetectionModel(object_detect, frame):
  return _runDetectionModel(object_detect, frame)

def runHandDetectionModel(hand_detect, frame):
  return _runDetectionModel(hand_detect, frame)

def runFaceDetectionModel(face_detect, frame):
  return _runDetectionModel(face_detect, frame)

def runPoseEstimationModel(pose_estimator, frame):
  results = pose_estimator.estimate(frame)
  image = results.draw_poses(frame)
  return frame, ""

def runSegmentationModel(segmentator, frame):
  results = segmentator.segment_image(frame)
  
  color_mask = segmentator.build_image_mask(results.class_map)
  blended_image = edgeiq.blend_images(frame, color_mask, alpha=0.5)
  return blended_image, ""

def _runClassifier(classifier, frame):
  results = classifier.classify_image(frame, confidence_level=0.5)
  text = ["Model: {}".format(classifier.model_id)]
  text.append(
          "Inference time: {:1.3f} s".format(results.duration))
  text.append("Inference:")
  for prediction in results.predictions:
      text.append("{}: {:2.2f}%".format(
          prediction.label, prediction.confidence * 100))
  return frame, text

def runGenderModel(gender_classifier, frame):
  return _runClassifier(gender_classifier, frame)

def runAgeModel(age_classifier, frame):
  return _runClassifier(age_classifier, frame)

def main(camera, use_streamer, server_addr, stream_fps):
    fps = edgeiq.FPS()
    try:
        streamer = None
        if use_streamer:
            streamer = edgeiq.Streamer().setup()
        else:
            streamer = CVClient(server_addr, stream_fps, sio).setup()

        with edgeiq.WebcamVideoStream(cam=camera) as video_stream:
            # Allow Webcam to warm up
            time.sleep(2.0)
            fps.start()
            prevModelChoice = None
            
            while True:
                if model_choice == "q":
                  print('Exiting program...')
                  if streamer is not None:
                    streamer.close()
                  fps.stop()
                  return
                model = models[model_choice]
                if model_choice != prevModelChoice and prevModelChoice is not None:
                  displayRuntimeStatistics(model)
                frame = video_stream.read()
                frame, text = runModel(model, frame, model_choice)
                streamer.send_data(frame, text) 
                fps.update()
                if streamer.check_exit():
                    break
                
    finally:
        if streamer is not None:
            streamer.close()
        fps.stop()
        print("elapsed time: {:.2f}".format(fps.get_elapsed_seconds()))
        print("approx. FPS: {:.2f}".format(fps.compute_fps()))
        print("Program Ending")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='alwaysAI Video Streamer')
    parser.add_argument(
            '--camera', type=int, default='0',
            help='The camera index to stream from.')
    parser.add_argument(
            '--use-streamer',  action='store_true',
            help='Use the embedded streamer instead of connecting to the server.')
    parser.add_argument(
            '--server-addr',  type=str, default='localhost',
            help='The IP address or hostname of the SocketIO server.')
    parser.add_argument(
            '--stream-fps',  type=float, default=20.0,
            help='The rate to send frames to the server.')
    parser.add_argument(
            '--model-choice',  type=str, default='object_detect',
            help='What kind of inference model to run on the camera stream.')
    args = parser.parse_args()
    model_choice = args.model_choice
    initializeModels()
    t = threading.Thread(target=main, args=(args.camera, args.use_streamer, args.server_addr, args.stream_fps))
    t.start()
    while True:
      inp = input("Change model to (press q to quit): ")
      
      
      if inp not in MODELS:
        print("Incorrect model: " + inp)
        print("Please enter one of ", MODELS)
      model_choice = inp
      if model_choice == "q":
        t.join()
        exit()
