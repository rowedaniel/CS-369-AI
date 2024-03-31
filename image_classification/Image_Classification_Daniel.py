import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

from random import sample, choice
from glob import glob
import time

import cv2 as cv
from PIL import Image

from torch import nn
import torchvision.models as models
from torchvision import transforms

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from skimage.feature import hog
from skimage.color import rgb2gray
from sklearn.decomposition import PCA


def get_raw_data(root_path, max_num_per_class=100):
    # split into subfolders based on class label
    subfolders = sorted(glob(root_path + "*"))
    label_names = [p.split("/")[-1] for p in subfolders]

    data = {
        "image_labels": [],
        "image_filenames": [],
        "images": [],
    }

    for i, subfolder in enumerate(subfolders):
        # get list of file paths for each subfolder
        file_paths = sorted(glob(subfolder + "/*.jpg"))
        for f in file_paths[:max_num_per_class]:
            img = cv.imread(f)
            data["image_labels"].append(i)
            data["image_filenames"].append(f)
            data["images"].append(img)

    return data, label_names


## declare how to create feature vectors


# prep a pretrained neural network (ignoring the final layer) to use for features
def slice_model(original_model, from_layer=None, to_layer=None):
    return nn.Sequential(*list(original_model.children())[from_layer:to_layer])


pretrained_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model_conv_features = slice_model(pretrained_model, to_layer=-1).to("cpu")
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def retype_image(img):
    if np.max(img) > 1:
        img = img.astype(np.uint8)
    else:
        img = (img * 255.0).astype(np.uint8)
    return img


def hog_from_img(img):
    if img.shape[0] != 100 or img.shape[1] != 100:
        img = cv.resize(img, (100, 100), interpolation=cv.INTER_CUBIC)
    feature_vector = hog(
        img,
        orientations=10,
        pixels_per_cell=(15, 15),
        cells_per_block=(4, 4),
        block_norm="L2",
        channel_axis=-1,
    )
    return feature_vector


def nn_emb_from_img(img):
    if img.shape[0] != 100 or img.shape[1] != 100:
        img = cv.resize(img, (100, 100), interpolation=cv.INTER_CUBIC)
    retyped_image = Image.fromarray(retype_image(img))
    preprocessed_img = preprocess(retyped_image)
    emb = (
        model_conv_features(preprocessed_img.unsqueeze(0).to("cpu"))
        .squeeze()
        .detach()
        .numpy()
    )
    return emb


def hist_from_img(img):
    feature_vector = np.histogram(rgb2gray(img), bins=16, density=True)[0]
    return feature_vector


def feature_vector_from_img(img):
    vec1 = nn_emb_from_img(img)
    vec2 = hog_from_img(img)
    vec3 = hist_from_img(img)
    feature_vector = np.hstack((vec1, vec2, vec3))
    return feature_vector


def rotate_image(img, angle):
    rows, cols, _ = img.shape
    img_centers = (75, 75)
    right_matrix = cv.getRotationMatrix2D(img_centers, angle, 1)
    return cv.warpAffine(img, right_matrix, (rows, cols), flags=cv.INTER_CUBIC)[
        10:-10, 10:-10
    ]


def extract_features(X_train_base, X_test_base, y_train_base, y_test_base):
    X_train_all = []
    y_train_all = []

    # data augmentation and preprocessing for train set
    for img_base, label in zip(X_train_base, y_train_base):
        img_flipped = cv.flip(img_base, 1)

        imgs_to_add = []
        for img in [img_base, img_flipped]:
            imgs_to_add.append(img)

            # if you want to further augment, rotate by 10 degrees left/right
            # for angle in (350, 10):
            #     imgs_to_add.append(rotate_image(img, angle))

        for img in imgs_to_add:
            feature_vector = feature_vector_from_img(img)

            X_train_all.append(feature_vector)
            y_train_all.append(label)

    # preprocessing for test set
    X_test_all = [feature_vector_from_img(img) for img in X_test_base]
    y_test_all = y_test_base

    return X_train_all, X_test_all, y_train_all, y_test_all


def main():
    # Get raw image data
    root_path = "./image_classification/Intel Training Dataset/"

    start_time = time.time()
    data, label_names = get_raw_data(
        root_path, max_num_per_class=8000
    )  # for time reasons, I advise only running this on 100-1000 images per class
    total_time = time.time() - start_time
    print(f"Time elapsed for getting raw data: {total_time}")

    # train/validation split
    images = np.array(data["images"])
    labels = np.array(data["image_labels"])

    # train/test split first, so we don't augment the test data
    X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(
        images, labels, test_size=0.2, stratify=labels, random_state=0
    )

    # extract features
    start_time = time.time()
    # put the nn in evaluation mode
    pretrained_model.eval()
    X_train_all, X_test_all, y_train_all, y_test_all = extract_features(
        X_train_base, X_test_base, y_train_base, y_test_base
    )
    X_train_all = np.array(X_train_all)
    y_train = np.array(y_train_all)
    X_test_all = np.array(X_test_all)
    y_test = np.array(y_test_all)
    total_time = time.time() - start_time
    print(f"Time elapsed for extracting features: {total_time}")

    # use principle componenet analysis to choose better features
    pca = PCA(n_components=min(1000, X_train_all.shape[0]))
    pca.fit(X_train_all)
    X_train = pca.transform(X_train_all)
    X_test = pca.transform(X_test_all)

    # train one model
    start_time = time.time()
    
    # Here we use SVC, but we also tried XGBoost
    clf = make_pipeline(StandardScaler(), SVC(gamma="scale", C=1, kernel="rbf"))
    # clf = make_pipeline(StandardScaler(), SVC(XGBClassifier()))
    clf.fit(X_train, y_train)
    total_time = time.time() - start_time
    print(f"Time elapsed for training one model: {total_time}")

    # report accuracy
    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)
    print(f"Overall train accuracy:")
    print(classification_report(y_train, train_pred))
    print(f"Overall test accuracy:")
    print(classification_report(y_test, test_pred))

    C_train = confusion_matrix(y_train, train_pred)
    ax1 = sn.heatmap(
        C_train,
        annot=True,
        xticklabels=label_names,
        yticklabels=label_names,
        cmap="Blues",
    )
    ax1.set(title="train results")

    plt.show()

    C_test = confusion_matrix(y_test, test_pred)
    ax2 = sn.heatmap(
        C_test,
        annot=True,
        xticklabels=label_names,
        yticklabels=label_names,
        cmap="Blues",
    )
    ax2.set(title="test results")

    plt.show()


def train_with_gridsearch(X_train, X_test, y_train, y_test, label_names):
    # train model w/ gridsearch

    parameters = {
        "kernel": ["rbf", "linear"],
        "C": [2**i for i in range(-3, 5)],
        "gamma": ["auto", "scale", *[2**i for i in range(-2, 3)]],
    }
    svc = SVC()
    gs = GridSearchCV(estimator=svc, param_grid=parameters, cv=5, verbose=2)
    gs.fit(X_train, y_train)

    # plot gridsearch results
    print(sorted(gs.cv_results_.items()))

    for kernel in set(gs.cv_results_["param_kernel"]):
        gamma_tested = parameters["gamma"]
        C_tested = parameters["C"]

        n_kernels = len(gamma_tested)
        bar_x = np.arange(len(C_tested))
        offset = 1 / (len(gamma_tested) + 1)

        fig, ax = plt.subplots(layout="constrained")

        for i, gamma in enumerate(gamma_tested):
            mask = (gs.cv_results_["param_gamma"] == gamma) & (
                gs.cv_results_["param_kernel"] == kernel
            )
            result_data = gs.cv_results_["mean_test_score"][mask]
            plt.bar(bar_x + i * offset, result_data, offset, label=gamma)

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel("score")
        ax.set_xticks(bar_x + offset, C_tested)
        ax.set_xlabel("C")
        ax.set_title(f"kernel = {kernel}")
        ax.legend()

        plt.show()

    # train one model with best params from gridsearch

    best_params = gs.best_params_
    print(best_params)
    clf = make_pipeline(
        StandardScaler(),
        SVC(
            kernel=best_params["kernel"],
            C=best_params["C"],
            gamma=best_params["gamma"],
        ),
    )
    clf.fit(X_train, y_train)

    # report accuracy of gridsearch-approved best model

    train_pred = clf.predict(X_train)
    test_pred = clf.predict(X_test)
    print(f"Overall train accuracy:")
    print(classification_report(y_train, train_pred))
    print(f"Overall test accuracy:")
    print(classification_report(y_test, test_pred))

    C = confusion_matrix(y_test, test_pred)
    sn.heatmap(
        C, annot=True, xticklabels=label_names, yticklabels=label_names, cmap="Blues"
    )


if __name__ == "__main__":
    main()
