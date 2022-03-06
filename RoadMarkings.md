**Обобщённый алгоритм**

- Предобработка данных в зависимости от качества съемки, фильтрация.
- Перевод в HSV, выделение белого и жёлтого цветов. 
```
### example code

hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) 

# binary mask for 'colors'
def get_lane_lines_mask(hsv_image, colors):
	masks = []
	    for color in colors:
	        if 'low_th' in color and 'high_th' in color:
	            mask = cv2.inRange(hsv_image, color['low_th'], color['high_th'])
	        	if 'kernel' in color:
                	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, color['kernel'])
            	masks.append(mask)
	return cv2.add(*masks)
    
WHITE_LINES = { 'low_th': gimp_to_opencv_hsv(0, 0, 80),
                'high_th': gimp_to_opencv_hsv(359, 10, 100) }

YELLOW_LINES = { 'low_th': gimp_to_opencv_hsv(35, 20, 30),
                 'high_th': gimp_to_opencv_hsv(65, 100, 100),
                 'kernel': np.ones((3,3),np.uint64)}
binary_mask = get_lane_lines_mask(hsv_image, [WHITE_LINES, YELLOW_LINES])

# apply mask
def draw_binary_mask(binary_mask, img):
    masked_image = np.zeros_like(img)
    for i in range(3): 
        masked_image[:,:,i] = binary_mask.copy()
    return masked_image

masked_image = draw_binary_mask(binary_mask, hsv_image)
```
- Есть смысл обрезать часть изображения (например, выше линии горизонта), где нет разметки, чтобы не обрабатывать ее.
- В некоторых алгоритмах применяется размытие по Гауссу.
```
cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
```
- Векторизация: детектор границ Кэнни -> преобразование Хафа для соединения границ в линии.
```
edges_mask = cv2.Canny(masked_img, low_threshold, high_threshold)
blank_image = np.zeros_like(image)
edges_img = draw_binary_mask(edges_mask, blank_image)

segments = cv2.HoughLinesP(edges_mask, rho, theta, threshold, np.array([]), 
                            min_line_length, max_line_gap)
```

- Выделение левой и правой линии разметки.
	- Для каждого сегмента, полученного преобразованием Хафа, строятся уравнения прямых
	- Линии делятся на правые и левые в соответствии с уклоном
	- Отсеиваются горизонтальные линии, линии выше горизонта и линии, которые слишком сильно отличаются по положению или уклону от разметки предыдущего кадра.
	- Все подходяшие сегменты аппроксимируются двумя полиномами 1-го порядка
	- Эта информация может использоваться для определения положения транспортного средства и правильности траектории движения по разметке

- Ожидается, что на следующих кадрах линии не сильно отклонятся от прошлого кадра, определение нужной полосы облегчается этим условием. Например, можно брать взвешенное среднее с последних нескольких кадров. Неверные линии могут учитываться на следующих шагах, но с меньшим коэффициентом, так как они могли нести полезную информацию.

Алгоритм работает хорошо на прямых линиях, возможно для перекрестков стоит попробовать аппроксимацию кривыми большего порядка

1) https://towardsdatascience.com/finding-lane-lines-simple-pipeline-for-lane-detection-d02b62e7572b

2) https://towardsdatascience.com/carnd-project-1-lane-lines-detection-a-complete-pipeline-6b815037d02c

3) https://docplayer.com/68750022-Algoritm-opredeleniya-sploshnyh-liniy-razmetki-dorozhnogo-polotna-vvedenie.html

4) https://medium.com/swlh/lane-finding-with-computer-vision-techniques-bad24828dbc0

5) https://www.researchgate.net/publication/345309351_Fast_Hough_Transform-Based_Road_Markings_Detection_For_Autonomous_Vehicle
