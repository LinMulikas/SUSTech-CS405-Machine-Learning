import cv2
import os
from train import processFiles, trainSVM
from detector import Detector

# Replace these with the directories containing your
# positive and negative sample images, respectively.

pos_dir = "./samples/vehicles"
neg_dir = "./samples/non-vehicles"

# Replace this with the path to your test video file.
video_file = "videos/test_video.mp4"


def finalExperment():
    feature_data = processFiles(
        pos_dir, neg_dir, 
        recurse=True,
        color_space="YCrCb", 
        channels= [0, 1, 2], 
        hog_features=True,
        hist_features=True, 
        spatial_features=True, 
        hog_lib="cv",
        size=(64,64), 
        pix_per_cell=(8,8), 
        cells_per_block=(2,2),
        hog_bins=20, 
        hist_bins=16, 
        spatial_size=(20,20)
        )

    classifier_data = trainSVM(feature_data=feature_data, C=1000)

    detector = Detector(
        init_size=(90,90), 
        x_overlap=0.7, 
        y_step=0.01,
        x_range=(0.02, 0.98), 
        y_range=(0.55, 0.89), 
        scale=1.3
        )
    
    detector.loadClassifier(classifier_data=classifier_data)

    cap = cv2.VideoCapture(video_file)
    detector.detectVideo(
        video_capture=cap, 
        num_frames=9, 
        threshold=120,
        draw_heatmap_size=0.3
        )


# def experiment2
#    ...

if __name__ == "__main__":
    finalExperment()
    # experiment2() may you need to try other parameters
    # experiment3 ...


