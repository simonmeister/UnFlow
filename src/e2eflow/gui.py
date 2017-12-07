import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


def display(results, image_names, title="Flow eval"):
    image_grids = []
    num_images = len(results[0])
    num_rows = len(results[0][0])
    num_cols = len(results)

    image_grids = []
    for i in range(num_images):
        image_grid = []
        for image_lists in results:
            image_grid.append(image_lists[i])
        image_grids.append(image_grid)

    fig = plt.figure(facecolor='grey')
    fig.suptitle(title)
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    imshow_images = []
    plt.subplots_adjust(wspace=0, hspace=0.03)

    imshow_image_lists = []
    for j, image_col in enumerate(image_grids[0]):
        imshow_images = []
        for i, t in enumerate(zip(image_names, image_col)):
            title, image = t
            ax = fig.add_subplot(num_rows, num_cols, i * num_cols + j + 1)
            if j == 0:
                ax.set_ylabel(title)
            ax.set_yticks([])
            ax.set_xticks([])
            if np.size(image, 3) == 1:
                imshow_images.append(ax.imshow(image[0, :, :, 0], "gray"))
            else:
                imshow_images.append(ax.imshow(image[0, :, :, :]))
        imshow_image_lists.append(imshow_images)

    def display_example(index):
        for j, image_col in enumerate(image_grids[int(index)]):
            imshow_images = imshow_image_lists[j]
            for im, image in zip(imshow_images, image_col):
                if np.size(image, 3) == 1:
                    im.set_data(image[0, :, :, 0])
                else:
                    im.set_data(image[0, :, :, :])
        plt.draw()

    current_index = 0

    next_button_ax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    next_button = Button(next_button_ax, 'next')
    prev_button_ax = fig.add_axes([0.7, 0.025, 0.1, 0.04])
    prev_button = Button(prev_button_ax, 'previous')
    slider_ax  = fig.add_axes([0.1, 0.025, 0.55, 0.04])
    slider = Slider(slider_ax, 'Page', 0, num_images - 1,
                    valinit=1, valfmt='%0.0f')

    def next_button_on_clicked(mouse_event):
        nonlocal current_index
        if current_index < num_images - 1:
            current_index += 1
            slider.set_val(current_index)

    def prev_button_on_clicked(mouse_event):
        nonlocal current_index
        if current_index > 0:
            current_index -= 1
            slider.set_val(current_index)

    def sliders_on_changed(val):
        nonlocal current_index
        current_index = val
        display_example(val)

    slider.on_changed(sliders_on_changed)
    prev_button.on_clicked(prev_button_on_clicked)
    next_button.on_clicked(next_button_on_clicked)

    plt.draw()
    plt.show()
