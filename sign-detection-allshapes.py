import math
from msilib.schema import TextStyle
import cv2 as cv2
import numpy as np
import os
import csv
import tensorflow as tf

IMG_WIDTH = 30
IMG_HEIGHT = 30

def main():

    # Rutas necesarias
    model_file = "model.h5" # Modelo entrenado con trafficSign-modelTrainer.py y el dataset GTSRB
    image_directory = "imagenes-entrada"
    result_directory = "resultados"
    category_names_file = "nombresCategorias.csv"
    category_names_dict = dict()

    # Carga del modelo previamente entrenado
    model = tf.keras.models.load_model(model_file)
    model.summary()

    # Lectura de csv con la relacion entre Id,NombreSeñal para etiquetar las imagenes
    with open(category_names_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Csv Column names are {", ".join(row)}')
                line_count += 1
            else:
                category_names_dict[row[0]] = row[1]
                line_count += 1

    print(f"Category Names: {category_names_dict}\n")

    images = list()

    # Leemos todas las imagenes del directorio de imagenes
    for image_name in os.listdir(image_directory):
        img = cv2.imread(os.path.join(image_directory, image_name))
        images.append((image_name,img))

        try:
            os.mkdir(os.path.join(result_directory, image_name))
        except OSError:
            print ("Creation of the directory %s failed" % os.path.join(result_directory, image_name))

        try:
            os.mkdir(os.path.join(result_directory, image_name, "regions"))
        except OSError:
            print ("Creation of the directory %s failed" % os.path.join(result_directory, image_name, "regions"))

    # Aplicamos el algoritmo para cada una de las imagenes leidas
    for img_tuple in images:
        img = img_tuple[1]
        img_name = img_tuple[0]
        resultPath = os.path.join(result_directory, img_name)

        # Pasamos la imagen a Gris
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(resultPath, "gray.png"), gray)

        # Aplicamos suavizado Gaussiano con kernel de 5x5, y desviacion tipica calculada a partir del tamaño del kernel
        gaussianBlur = cv2.GaussianBlur(gray, (5,5), 0)
        cv2.imwrite(os.path.join(resultPath, "gaussianBlur.png"), gaussianBlur)

        # Aplicamos el metodo de deteccion de bordes de Canny, con los thresholds entre 150 y 200
        canny = cv2.Canny(gaussianBlur, 150, 200)
        cv2.imwrite(os.path.join(resultPath, "canny.png"), canny)

        # Obtenemos los contornos de diversas formas a partir de la imagen con los bordes detectador por Canny
        contours,hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        all_regions_image = img.copy()
        signs_image = img.copy()

        rect_padding = 10

        # Almacenamos el area de la imagen
        image_area = len(img)*len(img[0])

        # Iteramos por cada uno de los contornos detectados en la imagen
        for i in range(1, len(contours)):

            # Si el area del contorno es menor que el 15% del area total de la imagen, lo obviamos
            if (cv2.contourArea(contours[i]) < 0.015*image_area):
                continue

            print(f"\nImage {img_name}, ROI={i}")

            # Aproximamos la curva de la imagen para que tenga un menor numero de vertices
            approx = cv2.approxPolyDP(contours[i], 0.01 * cv2.arcLength(contours[i], True), True)

            #Dibujamos la figura correspondiente en funcion del numero de vertices
            if len(approx)==3:
                # Dibujamos un triangulo verde
                cv2.drawContours(all_regions_image,[approx],0,(0,255,0),2)  
            elif len(approx)==4:
                # Dibujamos un cuadrado rojo
                cv2.drawContours(all_regions_image,[approx],0,(0,0,255),2)
            elif len(approx) > 12:
                # Dibujamos un circulo amarillo
                cv2.drawContours(all_regions_image,[contours[i]],0,(0,255,255),2)

            # Obtenemos el rectangulo que contiene al contorno
            x,y,w,h = cv2.boundingRect(approx)
            print(f"x: {x}, y: {y}, w: {w}, h: {h}")

            # Añadimos un padding al rectangulo
            x_pad = int(0.05*(x+w))
            y_pad = int(0.08*(y+h))

            # Calculamos las coordenadas correspondientes para evitar que se salga de la imagen
            lower_x_bound = x-x_pad if x-x_pad > 0 else 0
            upper_x_bound = x+w+x_pad if x+w+x_pad < len(img[0]) else len(img[0])-1
            lower_y_bound = y-y_pad if y-y_pad > 0 else 0
            upper_y_bound = y+h+y_pad if y+h+y_pad < len(img) else len(img)-1
            print(f"LowerX: {lower_x_bound}, UpperX: {upper_x_bound}, LowerY: {lower_y_bound}, UpperY: {upper_y_bound}")

            # Dibujamos el rectangulo en la imagen que contiene todas las regiones
            cv2.rectangle(all_regions_image,(lower_x_bound,lower_y_bound),(upper_x_bound,upper_y_bound),(0,0,0),2)

            # Extraemos la forma rectangular como una nueva imagen
            roi = img[lower_y_bound:upper_y_bound, lower_x_bound:upper_x_bound]
            cv2.imwrite(os.path.join(resultPath, "regions", str(i) + ".png"), roi)

            # Adaptamos la imagen de la region, y obtenemos el tensor resultado de la RN
            res_roi = cv2.resize(roi, dsize=(IMG_WIDTH, IMG_HEIGHT))
            model_input = np.expand_dims(res_roi, axis=0)
            result_tensor = model(model_input)

            # La id asignada sera aquella con mayor probabilidad dentro de la distribucion devuelta por el tensor
            sign_id = tf.argmax(result_tensor, 1)

            # Debug
            print(f"Resulting tensor: {result_tensor}")
            print(f"Resulting sign: {sign_id}")
            sorted_ids = tf.argsort(result_tensor,1,'DESCENDING')
            print(f"Sorted tensor: {sorted_ids}")
            #

            # Obtenemos el nombre de la señal a partir del csv
            sign_name = category_names_dict.get(str(sign_id.numpy()[0]))

            # Escribimos en una copia de la imagen la señal detectada, con el nombre correspondiente

            thickness = 1 if len(img) < 500 else 2
            desired_pixels = min(int(len(img)*0.05), 35)
            #print(f"Pixels: {desired_pixels}")
            font_scale = cv2.getFontScaleFromHeight(cv2.FONT_HERSHEY_SIMPLEX, desired_pixels, thickness)
            #print(f"Font_scale: {font_scale}")
            text_size, baseline = cv2.getTextSize(sign_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            #print(f"Text_size: {text_size}")

            textXPos = lower_x_bound-int(text_size[0]/3)
            textXPos = textXPos if textXPos+text_size[0] < len(img) else textXPos - (text_size[0] - (len(img) - textXPos))
            textXPos = textXPos if textXPos > 0 else 10

            textYPos = lower_y_bound-text_size[1]
            textYPos = textYPos if textYPos-text_size[1] > 0 else textYPos - (textYPos - text_size[1])

            textOrg = (textXPos, textYPos)
            #print(f"TextOrg: ({textOrg[0]}, {textOrg[1]})")

            cv2.rectangle(signs_image,(lower_x_bound,lower_y_bound),(upper_x_bound,upper_y_bound),(0,0,0),3)
            cv2.putText(signs_image, text=sign_name, org=textOrg, fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, color=(255, 0, 0), thickness=thickness, lineType=cv2.LINE_AA)

        # Escribimos las dos imagenes resultantes, una con las regiones detectadas, y otra con las señales ya reconocidas
        cv2.imwrite(os.path.join(resultPath, "with_regions.png"), all_regions_image)
        cv2.imwrite(os.path.join(resultPath, "with_labeled_signs.png"), signs_image)

        # DEBUG
        # cv2.imshow('detected circles',all_regions_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()