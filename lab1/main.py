import cv2
import numpy as np
import matplotlib.pyplot as plt

def fft(img):       # prebacuje sliku u frekventni domen
    img_fft = np.fft.fft2(img)
    img_fft = np.fft.fftshift(img_fft)
    return img_fft

def inverse_fft(magnitude_log, complex_moduo_1):  # vraca sliku u prostorni domen
    img_fft = complex_moduo_1 * np.exp(magnitude_log)
    img_filtered = np.abs(np.fft.ifft2(img_fft))

    return img_filtered


def fft_noise_removing(img, center):
    img_fft = fft(img)
    img_fft_mag = np.abs(img_fft)

    plt.imshow(img_fft_mag)  #amplituda/,magnituda pre uklanjanja suma

    description = "Magnituda pre filtriranja:"
    plt.title(description)
    plt.show()

    img_mag_1 = img_fft / img_fft_mag
    img_fft_log = np.log(img_fft_mag)

    img_fft_log[center[0] - 25, center[1] - 25] = 0
    img_fft_log[center[0] + 25, center[1] + 25] = 0

    img_fft_log[center[0] - 100, center[1] + 100] = 0
    img_fft_log[center[0] + 100, center[1] - 100] = 0

    img_filtered = inverse_fft(img_fft_log, img_mag_1)

    img_fft_mag2 = np.exp(img_fft_log)  #magnituda nakon uklanjanja suma
    plt.imshow(img_fft_mag2)

    description = "Magnituda nakon filtriranja:"
    plt.title(description)
    plt.show()

    return img_filtered




if __name__ == '__main__':
    img = cv2.imread("../slika_3.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    center = (256, 256)
    #radius = 50

    plt.imshow(img, cmap='gray')
    description = "Pocetna slika:"
    plt.title(description)
    plt.show()  # pocetna slika

    img_final = fft_noise_removing(img, center)  # uklanjanje periodicnog suma

    plt.imshow(img_final, cmap='gray')
    description = "Krajnja slika:"
    plt.title(description)
    plt.show()

    cv2.imwrite('../output_image.png', img_final) #cuva finalnu sliku