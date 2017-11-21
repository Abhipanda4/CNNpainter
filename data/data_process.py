import numpy as np
import os
import shutil
import csv
from PIL import Image
from PIL import ImageFile
from collections import defaultdict
import numpy as np
import operator


INIT_DIR = "/home/abhipanda/CNNpainter/data"
WORKING_DIR = "/home/abhipanda/CNNpainter/data/train"
SAVE_DIR = "/home/abhipanda/CNNpainter/data/PROCESSED_DATA"
TRAIN_DIR = os.path.join(SAVE_DIR, "TRAIN")
VALIDATION_DIR = os.path.join(SAVE_DIR, "VALIDATION")
TEST_DIR = os.path.join(INIT_DIR, "TEST")
TEST_INIT = os.path.join(INIT_DIR, "test")
ImageFile.LOAD_TRUNCATED_IMAGES = True

class DataProcessing(object):
    """
    Contains functions related to separating images on the basis of their
    painters and keeping them in separate directories
    """

    def __init__(self, INPUT_DIM, PROCESSING_DIM, todelete):
        self.PROCESSING_DIM = PROCESSING_DIM
        self.INPUT_DIM = INPUT_DIM
        if todelete:
            shutil.rmtree(SAVE_DIR)
        if not os.path.exists(SAVE_DIR):
            os.mkdir(SAVE_DIR)
        if not os.path.exists(TRAIN_DIR):
            os.mkdir(TRAIN_DIR)
        if not os.path.exists(VALIDATION_DIR):
            os.mkdir(VALIDATION_DIR)
        if not os.path.exists(TEST_DIR):
            os.mkdir(TEST_DIR)



    def __painting_by_artist(self):
        """
        Returns an array  of tuples. The tuple is of the form:
        (filename, artist)
        """
        file = os.path.join(INIT_DIR, "train_info.csv")
        with open(file) as f:
            reader = csv.DictReader(f, delimiter=",")
            data = []
            for row in reader:
                data.append((row['filename'], row['artist']))
        return data


    def __processImage(self, f):
        """
        Returns a center cropped image of the original image.
        It first resizes the image to the required dimension along the
        smallest dimension and then cuts off the other aspect.

        Takes the image file path as parameter
        """
        try:
            imgobj = Image.open(f).convert('RGB')
        except:
            return None
        w, h = imgobj.size
        if w < h:
            # reduce width to required dimension and adjust height accordingly
            new_h = int(h * self.PROCESSING_DIM / w)
            resizedImg = imgobj.resize((self.PROCESSING_DIM, new_h))

            y_start = int(new_h / 2 - self.PROCESSING_DIM / 2)
            processedImage = resizedImg.crop((0, y_start, self.PROCESSING_DIM, y_start + self.PROCESSING_DIM))

        else:
            # reduce height to required dimension and adjust width accordingly
            new_w = int(w * self.PROCESSING_DIM / h)
            resizedImg = imgobj.resize((new_w, self.PROCESSING_DIM))

            x_start = int(new_w / 2 - self.PROCESSING_DIM / 2)
            processedImage = resizedImg.crop((x_start, 0, x_start + self.PROCESSING_DIM, self.PROCESSING_DIM))

        return processedImage


    def __randomCrop(self, img):
        """
        Returns a random INPUT_DIM X INPUT_DIM image crop from the processed image
        No boundary exceed allowed
        """
        limit = self.PROCESSING_DIM - self.INPUT_DIM
        # pick 2 random integers less than this limit as the origin of the cropped image
        x_start = np.random.randint(limit)
        y_start = np.random.randint(limit)
        return img.crop((x_start, y_start, x_start + self.INPUT_DIM, y_start + self.INPUT_DIM))


    def __applyRotations(self, img):
        """
        Returns a list of images rotated by various angles:
        0, 90, 180, 270, mirror image and water image
        """
        res = [self.__randomCrop(img)]
        rotations = [
                Image.FLIP_LEFT_RIGHT,
                Image.FLIP_TOP_BOTTOM,
                Image.ROTATE_90,
                Image.ROTATE_180,
                Image.ROTATE_270,
                Image.TRANSPOSE
                ]
        for j in range(len(rotations)):
            i = self.__randomCrop(img)
            tmp = i.transpose(j)
            res.append(tmp)
        return res


    def arrange_by_artists(self, isTest):
        """
        Splits the image set into training and validation directories.
        Augments data by rotating the original images
        Final size of image is INPUT_DIM x INPUT_DIM
        """
        batch_size = 100
        info = self.__painting_by_artist()
        np.random.shuffle(info)
        for i in range(0, len(info), batch_size):
            imagelist = self.__augmented_images(info[i : i + batch_size], i)
            np.random.shuffle(imagelist)
            num_train = int(5/6 * len(imagelist))
            self.__save_to_dir(imagelist[:num_train], "train", TRAIN_DIR)
            self.__save_to_dir(imagelist[num_train:], "validation", VALIDATION_DIR)
            print("Batch Completed.")


    def test(self, k):
        self.__topK_train(k)


    def __save_to_dir(self, imagelist, prefix, PATH):
        """
        Save the images into appropriate directories. For keras,
        the directory structure should be:
            ---> Train:
                |_ Class 1:
                     |_img1.jpg
                     |_img2.jpg
                     |_img3.jpg
                     |_img4.jpg
                |_ Class 2:
                     |_img1.jpg
                     |_img2.jpg
                     |_img3.jpg
                     |_img4.jpg
                |_ Class 3:
                     |_img1.jpg
                     |_img2.jpg
                     |_img3.jpg
                     |_img4.jpg
            ---> V alidation:
                |_ Class 1:
                     |_img1.jpg
                     |_img2.jpg
                     |_img3.jpg
                     |_img4.jpg
                |_ Class 2:
                     |_img1.jpg
                     |_img2.jpg
                     |_img3.jpg
                     |_img4.jpg
                |_ Class 3:
                    |_img1.jpg
                    |_img2.jpg
                    |_img3.jpg
                    |_img4.jpg
        """
        for pair in imagelist:
            directory = os.path.join(PATH, pair[1])
            if not os.path.exists(directory):
                os.mkdir(directory)
            filename = prefix + pair[2]
            pair[0].save(os.path.join(directory, filename))
            print("Saved " + os.path.join(directory, filename))


    def __augmented_images(self, info, start):
        """
        Function to return a list of all the transformed images
        The list elements are tuples of the form:
        (image object, artistID, filename)
        """
        count = start
        final_img_to_save = []
        for pair in info:
            processedImage = self.__processImage(os.path.join(WORKING_DIR, pair[0]))
            if processedImage == None:
                continue
            # translation is not that important since CNNs are resistant to image translations
            rotatedImages = self.__applyRotations(processedImage)

            rotCount = 1
            for img in rotatedImages:
                filename = str(count) + "_" + str(rotCount) + ".jpg"
                # img.save(os.path.join(directory, filename))
                final_img_to_save.append((img, pair[1], filename))
                rotCount += 1

            print("Augmenting image: {:05}".format(count))
            count += 1
        return final_img_to_save


    def __topK_train(self, k):
        """
        Returns a list of top K artists in training set in the form of their IDs
        """
        f = open("train_info.csv")
        reader = csv.DictReader(f, delimiter=",")
        artists = []
        for line in reader:
            artists.append(line['artist'])
        freqs = defaultdict(int)
        for artist in artists:
            freqs[artist] += 1

        sorted_freqs = sorted(freqs.items(), key=operator.itemgetter(1))
        final_list = list(reversed(sorted_freqs))
        res = []
        for pair in final_list[:k]:
            res.append(pair[0])
            if not os.path.exists(os.path.join(TEST_DIR, pair[0])):
                os.mkdir(os.path.join(TEST_DIR, pair[0]))
        return res


    def __test_and_train(self):
        """
        returns a list of tuples. Each tuple consists of the artist name and filename
        of painting. The artist is such that his/her painting is also present in the
        training set
        """
        f = open("all_data_info.csv")
        reader = csv.DictReader(f, delimiter=",")
        data = []
        for line in reader:
            if line['artist_group'] == "train_and_test" and line["in_train"] == "False":
                # the img's artist is in training set
                # but the img is in test set only
                data.append((line['artist'], line['new_filename']))

        return data


    def __ID_to_name_mapping(self):
        f1 = open("train_info.csv")
        f2 = open("all_data_info.csv")

        reader1 = csv.DictReader(f1, delimiter=",")
        reader2 = csv.DictReader(f2, delimiter=",")

        # create dict: id -> name
        dict = {}
        for line1 in reader1:
            id = line1['artist']
            work = line1['filename']
            for line2 in reader2:
                if line2['new_filename'] == work:
                    dict[id] = line2['artist']
                    break

        return dict


    def create_test_images(self, k):
        """
        args: k -> Number of top artists to consider from the training set
        Keeps top k artists from the training set in the test set
        """
        train_artists = self.__topK_train(k)
        available_test_artists = self.__test_and_train()
        idtoname = self.__ID_to_name_mapping()
        for artist_id in train_artists:
            # scan all his/her images in the test set
            # if found, create folder accordingly
            artist_name = idtoname[artist_id]
            for name, file in available_test_artists:
                if name == artist_name:
                    filepath = os.path.join(TEST_DIR, artist_id)
                    if not os.path.exists(filepath):
                        os.mkdir(filepath)
                    # shutil.copyfile(os.path.join(TEST_INIT, file), os.path.join(filepath, file))
                    processedImage = self.__processImage(os.path.join(TEST_INIT, file))
                    print("Saving test_", file, "by artist: ", name)
                    processedImage.save(os.path.join(filepath, "test_" + file))

