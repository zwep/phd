from pdf2image import convert_from_path
import numpy as np

ddata = '/home/bugger/Documents/paper/thesis/Thesis/main.pdf'
images = convert_from_path(ddata)
# Collect all the hue and saturation of the pages...
hsv_list = []
for i_page, image in enumerate(images):
    img = np.array(image.convert('HSV'))
    hsv_sum = img.sum(0).sum(0)
    hsv_list.append((i_page, hsv_sum))

# Determine which pages are color and which are not
color_list = []
for i_page, i_hsv in hsv_list:
    if i_hsv[0] == 0 and i_hsv[1] == 0:
        pass
    else:
        color_list.append(i_page)

# Correct for python and the frontmatter
introduction_pages = 10
python_correction = 1
print(', '.join([str(x + python_correction) for x in color_list]))
print(', '.join([str(x - introduction_pages + python_correction) for x in color_list]))

print(','.join([str(x + 1) for x in color_list]))

print(', '.join([str(x - introduction_pages + python_correction ) for x in color_list]))