import numpy

import face_recognition
# from utils import detector_utils
import cv2
import datetime
import argparse
import pickle

from hand_recognition.utils import detector_utils

is_debug = True
detection_graph, sess = detector_utils.load_inference_graph()


def calculate_fps(num_frames):
    num_frames += 1
    elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
    fps = num_frames / elapsed_time
    return fps, elapsed_time, num_frames


def write_array_to_file(array):
    text_file = open("Output.txt", "w")
    text_file.write('\n'.join([''.join(['{:4}'.format(item) for item in row])
                               for row in array]))
    text_file.close()


if __name__ == '__main__':
    word = "everything"
    parser = argparse.ArgumentParser()
    parser.add_argument('-sth', '--scorethreshold', dest='score_thresh', type=float, default=0.39,
                        help='Score threshold for displaying bounding boxes')
    parser.add_argument('-fps', '--fps', dest='fps', type=int, default=1,
                        help='Show FPS on detection/display visualization')
    parser.add_argument('-src', '--source', dest='video_source', default='videos/' + word + '.mp4',
                        help='Device index of the camera.')
    parser.add_argument('-ds', '--display', dest='display', type=int, default=1,
                        help='Display the detected images using OpenCV. This reduces FPS')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_source)

    start_time = datetime.datetime.now()
    num_frames = 0
    frame_width, frame_height = (cap.get(3), cap.get(4))
    matrix = numpy.zeros((int(round(frame_height)), int(round(frame_width)))).astype(int).astype(str)

    # max number of hands we want to detect/track
    num_hands_detect = 2

    if is_debug:
        cv2.namedWindow('Single-Threaded Detection', cv2.WINDOW_NORMAL)

    # ********************** Face Detection Preparation **********************
    # Load a sample picture and learn how to recognize it.
    obama_image = face_recognition.load_image_file("face_recognition/obama.jpg")
    obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

    # Load a second sample picture and learn how to recognize it.
    biden_image = face_recognition.load_image_file("face_recognition/biden.jpg")
    biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

    # Create arrays of known face encodings and their names
    known_face_encodings = [obama_face_encoding, biden_face_encoding]
    known_face_names = ["Barack Obama", "Joe Biden"]

    # Initialize some variables
    max_num_of_hands = 0
    left = -1
    top = -1
    right = -1
    bottom = -1
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    is_face_detected = False
    points = list()

    # ********************** Face Detection Preparation **********************

    ret, frame = cap.read()
    while ret:
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        frame = cv2.flip(frame, 1)
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except:
            print("Error converting to RGB")

        if not is_face_detected:
            # Only process every other frame of video to save time
            if process_this_frame:
                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # If a match was found in known_face_encodings, just use the first one.
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]

                face_names.append(name)

                process_this_frame = not process_this_frame

            for (top, right, bottom, left), name in zip(face_locations, face_names):
                is_face_detected = True

                with open('face.pkl', 'wb') as f:
                    pickle.dump([str(top), str(right), str(bottom), str(left)], f)

                for x in range(left, right):
                    for y in range(top, bottom):
                        matrix[y][x] = "f"

        # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
        # while scores contains the confidence for each of these boxes.
        # Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score threshold)
        boxes, scores = detector_utils.detect_objects(frame, detection_graph, sess)

        # draw bounding boxes on frame
        matrix, num_of_hands_detected, _points = detector_utils.draw_box_on_image(num_hands_detect, args.score_thresh,
                                                                                  scores,
                                                                                  boxes, frame_width, frame_height,
                                                                                  frame,
                                                                                  matrix, num_frames, right, left)

        if len(_points) > 0:
            for x in (0,len(_points) - 1):
                points.append(_points[x])

        if max_num_of_hands < num_of_hands_detected:
            max_num_of_hands = num_of_hands_detected

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Calculate Frames per second (FPS)
        fps, elapsed_time, num_frames = calculate_fps(num_frames)

        if is_debug:
            if args.display > 0:
                if args.fps > 0:
                    detector_utils.draw_fps_on_image("FPS : " + str(int(fps)), frame)

                cv2.imshow('Single-Threaded Detection', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            cap.release()
            break
        # else:
        # print("frames processed: ", num_frames, "elapsed time: ", elapsed_time, "fps: ", str(int(fps)))

        ret, frame = cap.read()

if len(points) > 0:
    with open('max_hands.pkl', 'wb') as f:
        pickle.dump(max_num_of_hands, f)
    numpy.save('array_np.npy', matrix)  # .npy extension is added if not given
    numpy.save('data_words_arrays/array_' + word + '.npy', points)  # .npy extension is added if not given
    write_array_to_file(matrix)
