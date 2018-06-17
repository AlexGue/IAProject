import os
import cv2
import math
from matplotlib import pyplot as plt
import numpy as np
import time
from threading import Thread


def main(INPUTPATH, K, T, H, it, n, OUTPUTPATH):
    INPUTPATH = "C:/Users/Alejandro/Desktop/Universidad/TERCERO/IA/prueba2"
    OUTPUTPATH = "C:/Users/Alejandro/Desktop/Universidad/TERCERO/IA/results"
    image_names = os.listdir(INPUTPATH)
    image_names.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    print('Image names:')
    print(image_names)
    print('-------------------')
    image_histr = calc_histr(INPUTPATH, T, image_names, H)
    print('Calculado histogramas de cada imagen')
    print('-------------------')
    centroids = k_means(K, image_histr[0], 0, H, it, n)
    keyframes = write_keyframes(image_histr[1], image_histr[0], centroids, OUTPUTPATH)
    write_video(image_names, keyframes, INPUTPATH, OUTPUTPATH, 25)


def calc_histr(inputpath,T,image_names,H):
    image_histr = []
    name_image = dict()
    lastPercent = 0
    for i in range(0, len(image_names), T):
        if lastPercent != math.trunc(i/len(image_names)*100):
            lastPercent = math.trunc(i/len(image_names)*100)
            print('Calculando histogramas: ' + str(lastPercent) + '%')
        input_path = os.path.join(inputpath, image_names[i])

        img = cv2.imread(input_path)
        # cv2.imshow('scene', img)
        # width, heigth, chanels = img.shape
        color = ('b', 'g', 'r')
        name_image[image_names[i]] = img

        histr = dict()
        for j, col in enumerate(color):
            histr[col] = list(map(lambda x: x[0], cv2.calcHist([img], [j], None, [H], [0, H])))
            #plt.plot(histr[col])
        #plt.title('IMAGE' + image_names[i])
        #plt.show()
        #     plt.plot(histr[col], color=col)
        #     plt.xlim([0, H])
        #
        # plt.show()
        # print(str(histr) + ' ' + str(i))
        image_histr.append((image_names[i], histr))

    return [dict(image_histr), name_image]


def k_means(K, image_histr, size, H, iterations, n, centroids = None):

    color = ('b', 'g', 'r')
    last_centroids = None
    for it in range(0, iterations):
        if centroids is None:
            centroids = dict()
            s = math.trunc(len(image_histr)/K)
            for i in range(0, K):
                centroids[i] = list(image_histr.values())[i*s]
                for col in color:
                    plt.plot(centroids[i][col])
                plt.show()

        classified_images = {k:[] for k, k_histr in centroids.items()}
        print('Iteration number: '+str(it))
        for name, img_histr in image_histr.items():
            min_histr = []
            min_k = -1
            min_value = float('inf')
            for k, k_histr in centroids.items():
                value_acumulated = 0
                for j, col in enumerate(color):
                    value_acumulated += sum(np.sqrt(np.power([x1 - x2 for (x1, x2) in zip(img_histr[col], k_histr[col])], 2)))

                if value_acumulated < min_value:
                    min_value = value_acumulated
                    min_k = k
                    min_histr = k_histr
            classified_images[min_k].append(name)
        print('Classified images: ' + str(classified_images))
        newCentroids = dict()
        for k, classes in classified_images.items():
            if classes:
                histr = None
                for name in classified_images[k]:
                    if histr is None:
                        histr = {key: [int(values)/len(classified_images[k]) for values in image_histr[name][key]] for key in image_histr[name]}
                    else:
                        histr = {key_h: [sum(x)/len(classified_images[k]) for x in zip(*[histr[key_h], image_histr[name][key_h]])] for key_h in histr}
                newCentroids[k] = histr
            else:
                newCentroids[k] = centroids[k]
        print('New centroids : '+ str(centroids))
        if last_centroids is None:
            last_centroids = centroids
        else:
            if last_centroids.values() == centroids.values():
                break
            else:
                last_centroids = centroids
    return centroids

def write_keyframes(images, histograms, centroids, output):

    image_names = os.listdir(output)
    for name in image_names:
        os.remove(output+'/'+name)
    color = ('b', 'g', 'r')
    keyframes = dict()

    for i, k_histr in centroids.items():
        min_value = float('inf')
        for name, histr in histograms.items():
            value_acumulated = 0
            for j, col in enumerate(color):
                value_acumulated += sum(np.sqrt(np.power([x1 - x2 for (x1, x2) in zip(histr[col], k_histr[col])], 2)))
            if value_acumulated < min_value:
                min_value = value_acumulated
                keyframes[i] = name
    print('Writed images: ' + str(keyframes))
    for i, name in keyframes.items():
        cv2.imwrite(output+'/'+name, images[name])
    return keyframes

def write_video(images, keyframes, input, output, amount):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(output + '/result.avi', fourcc, 30.0, (2688, 1520))
    for k in keyframes.values():
        index = images.index(str(k))
        upIndex = index + amount
        downIndex = index - amount
        if (upIndex) > len(images):
            upIndex = len(images)
        if (downIndex < 0):
            downIndex = 0
        for image in images[downIndex:upIndex]:
            print('writing:' + image)
            video.write(cv2.imread(input + '/' + image))
    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(None, 3, 1, 256, 25, 3, None)
