import numpy as np
import cv2
import tqdm
import os
import sys
# color of different clusters
GBR = [[0, 0, 255],
       [0, 128, 255],
       [255, 0, 0],
       [128, 0, 128],
       [255, 0, 255]]

# path configuration
input_path = os.path.abspath('/Users/mulikas/Codes/CS405 ML/Lab/Lab11.K-means clustering')
output_path = os.path.join(os.path.abspath('/Users/mulikas/Codes/CS405 ML/Lab/Lab11.K-means clustering'), 'output')
if not os.path.exists(output_path):
    os.makedirs(output_path)
    
    
def kmeans(data: np.ndarray, n_cl: int):
    """
        K-means

    :param data:    original data
    :param n_cl:    number of classes
    :param seeds:   seeds
    :return:        new labels and new seeds
    """
    n_samples, channel = data.shape
    
    # TODO: firstly you should init centroids by a certain strategy
    args = np.random.choice(range(0, n_samples), n_cl, replace = False)
    centers = data[args]
    
    old_labels = np.zeros((n_samples,))
    while True:
        # TODO: calc distance between samples and centroids
        distance = np.ndarray((n_samples, n_cl))
        for i in range(0, n_samples):
            distance[i] = np.linalg.norm(centers - data[i], axis=1)
        
            
        # TODO: classify samples
        new_labels = np.argmin(distance, axis=1)

        # TODO: update centroids
        for i in range(0, n_cl):
            centers[i] = np.mean(data[np.where(new_labels == i)])
            
        
        if np.all(new_labels == old_labels):            
            break
        old_labels = new_labels

    return old_labels


def detect(video, n_cl=2):
    # load video, get number of frames and get shape of frame
    cap = cv2.VideoCapture(video)
    fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # instantiate a video writer
    video_writer = cv2.VideoWriter(
        os.path.join(output_path, "result_with_%dclz.mp4" % n_cl),
        cv2.VideoWriter_fourcc(*'mp4v'),
        (fps / 10),
        size,
        isColor=True
    )

    # initialize frame and seeds
    ret, frame = cap.read()
 

    print("Begin clustering with %d classes:" % n_cl)
    bar = tqdm.tqdm(total=fps)  # progress bar
    while ret:
        frame = np.float32(frame)
        h, w, c = frame.shape

        # k-means
        data = frame.reshape((h * w, c))
        labels = kmeans(data, n_cl=n_cl)

        # give different cluster different colors
        new_frame = np.zeros((h * w, c))
        # TODO: dye pixels with colors
        new_frame = new_frame.reshape((h, w, c)).astype("uint8")
        video_writer.write(new_frame)

        ret, frame = cap.read()
        bar.update()

    # release resources
    video_writer.release()
    cap.release()
    cv2.destroyAllWindows()


video_sample = os.path.join(input_path, "road_video.MOV")
detect(video_sample, n_cl=2)