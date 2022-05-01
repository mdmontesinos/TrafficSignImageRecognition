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
    model_file = "model.h5"
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

        # Aplicamos la transformacion de Hough para obtener figuras circulares, sobre la imagen con la deteccion de bordes
        detected_circles = cv2.HoughCircles(canny, 
                        cv2.HOUGH_GRADIENT, 1, 1, param1 = 50,
                    param2 = 50, minRadius = 0, maxRadius = 150)

        all_regions_image = img.copy()
        labeled_sign_image = img.copy()
        
        # Si se ha detectado algun circulo, continua el algoritmo
        if detected_circles is not None:
        
            # Array con los circulos detectados en la imagen
            detected_circles = np.uint16(np.around(detected_circles))
            # Nº de pixeles que se añaden de padding al extraer el fragmento de la imagen
            rectangle_padding = 30
            # Señal con mayor probabilidad (0: prob, 1: point, 2: sign_id)
            best_estimation = [0]

            # Para cada region detectada
            for i in range(0, len(detected_circles)):
                pt = detected_circles[0, i]
                # Obtenemos las coordenadas (a,b) del centro y el radio del circulo
                a, b, r = pt[0], pt[1], pt[2]

                # Calculamos las coordenadas correspondientes para obtener una forma rectangular alrededor de la region circular
                lower_x_bound = b-r-rectangle_padding if b-r-rectangle_padding > 0 else 0
                upper_x_bound = b+r+rectangle_padding if b+r+rectangle_padding < len(img) else len(img)-1
                lower_y_bound = a-r-rectangle_padding if a-r-rectangle_padding > 0 else 0
                upper_y_bound = a+r+rectangle_padding if a+r+rectangle_padding < len(img[0]) else len(img[0])-1
                #print(f"LowerX: {lower_x_bound}, UpperX: {upper_x_bound}, LowerY: {lower_y_bound}, UpperY: {upper_y_bound}")

                # Extraemos la forma rectangular como una nueva imagen
                roi = img[lower_x_bound:upper_x_bound, lower_y_bound:upper_y_bound]
                cv2.imwrite(os.path.join(resultPath, "regions", str(i) + ".png"), roi)
        
                # Dibujamos la circunferencia y centro del circulo en una copia de la imagen original,
                # donde apareceran todas las regiones circulares detectadas (incluyendo las falsas regiones).
                cv2.circle(all_regions_image, (a, b), r, (0, 255, 0), 4)
                cv2.circle(all_regions_image, (a, b), 1, (0, 0, 255), 3)

                # DEBUG
                # cv2.imshow('detected regions',all_regions_image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

                # Adaptamos la imagen de la region, y obtenemos el tensor resultado de la RN
                res_roi = cv2.resize(roi, dsize=(IMG_WIDTH, IMG_HEIGHT))
                model_input = np.expand_dims(res_roi, axis=0)
                result_tensor = model(model_input)

                # La id asignada sera aquella con mayor probabilidad dentro de la distribucion devuelta por el tensor
                sign_id = tf.argmax(result_tensor, 1)
                probability = result_tensor[0, sign_id.numpy()[0]]

                # Debug
                print(f"\nImage {img_name}, ROI={i}")
                print(f"Resulting tensor: {result_tensor}")
                print(f"Resulting sign: {sign_id}")
                print(f"Probability: {probability}")
                sorted_ids = tf.argsort(result_tensor,1,'DESCENDING')
                print(f"Sorted tensor: {sorted_ids}")
                #

                # De las regiones detectadas, nos quedaremos unicamente con aquella que ha sido identificada
                # por la red neuronal con mayor probabilidad, para tratar de descartar las falsas regiones
                if (probability > best_estimation[0]):
                    best_estimation = [probability, pt, sign_id.numpy()[0]]

            # Escribimos la imagen que contiene todas las regiones
            cv2.imwrite(os.path.join(resultPath, "with_regions.png"), all_regions_image)

            # Para la señal resultante, de mayor probabilidad,

            # obtenemos el nombre de la señal a partir del csv y
            # la informacion sobre el circulo que la contiene
            sign_name = category_names_dict.get(str(best_estimation[2]))
            pt = best_estimation[1]
            a, b, r = pt[0], pt[1], pt[2]

            # Se escribe en una copia de la imagen original unicamente la señal identificada finalmente

            thickness = 1 if len(img) < 500 else 2
            desired_pixels = min(int(len(img)*0.05), 35)
            #print(f"Pixels: {desired_pixels}")
            font_scale = cv2.getFontScaleFromHeight(cv2.FONT_HERSHEY_SIMPLEX, desired_pixels, thickness)
            #print(f"Font_scale: {font_scale}")
            text_size, baseline = cv2.getTextSize(sign_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            #print(f"Text_size: {text_size}")

            textXPos = a-r-int(text_size[0]/3)
            textXPos = textXPos if textXPos+text_size[0] < len(img) else textXPos - (text_size[0] - (len(img) - textXPos))
            textXPos = textXPos if textXPos > 0 else 10

            textYPos = b-r-text_size[1]
            textYPos = textYPos if textYPos-text_size[1] > 0 else textYPos - (textYPos - text_size[1])

            textOrg = (textXPos, textYPos)
            #print(f"TextOrg: ({textOrg[0]}, {textOrg[1]})")

            cv2.circle(labeled_sign_image, (a, b), r, (0, 255, 0), 4)
            cv2.circle(labeled_sign_image, (a, b), 1, (0, 0, 255), 3)
            cv2.putText(labeled_sign_image, text=sign_name, org=textOrg, fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, color=(255, 0, 0), thickness=thickness, lineType=cv2.LINE_AA)

            cv2.imwrite(os.path.join(resultPath, "with_labeled_sign.png"), labeled_sign_image)

if __name__ == "__main__":
    main()