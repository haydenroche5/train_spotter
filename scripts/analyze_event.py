import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from PIL import Image
import numpy as np

event_number = 0
moment_idx = 0
event_dir_path = '/home/hayden/cohub/events/{}/'.format(event_number)
image_dir_path = os.path.join(event_dir_path, 'images')
moments_file_path = os.path.join(event_dir_path, 'moments.pickle')

with open(moments_file_path, 'rb') as moments_file:
    moments = pickle.load(moments_file)
print("# moments: {}".format(len(moments)))

moment_idxs = np.arange(len(moments))
pred_values = np.array([moment['prediction_value'] for moment in moments])

fig = plt.figure()
fig.set_size_inches(10, 8)
pred_plot = fig.add_axes([0.1, 0.7, 0.8, 0.25])
pred_plot.set_xlabel('Moment')
pred_plot.set_ylabel('Prediction Value')
img_plot = fig.add_axes([0.1, 0.4, 0.8, 0.25])
img_plot.set_axis_off()

scatter = pred_plot.scatter(moment_idxs, pred_values, s=0.5)

annot = pred_plot.annotate("",
                           xy=(0, 0),
                           xytext=(20, 20),
                           textcoords="offset points",
                           bbox=dict(boxstyle="round", fc="w"),
                           arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)


def update_annot(ind):

    pos = scatter.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    text = "{}, {}".format(int(pos[0]), pos[1])
    annot.set_text(text)
    # annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
    # annot.get_bbox_patch().set_alpha(0.4)


def hover(event):
    vis = annot.get_visible()
    if event.inaxes == pred_plot:
        cont, ind = scatter.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()


fig.canvas.mpl_connect("motion_notify_event", hover)

# def next_moment():
#     global event_number
#     global moment_counter

#     if moment_counter >= len(moments):
#         return

#     moment_idxs.append(moment_counter)
#     pred_value = moments[moment_counter]['prediction_value']
#     pred_values.append(pred_value)
#     moment_counter += 1

#     img_path = os.path.join(
#         '/home/hayden/cohub/events/{}/images/{}.jpg'.format(
#             event_number, moment_counter))
#     img = Image.open(img_path)
#     img_plot.imshow(img)

#     # pred_line.set_xdata(moment_idxs)
#     # pred_line.set_ydata(pred_values)
#     pred_line, = pred_plot.plot(
#         moment_idxs, pred_values,
#         'b-')  # Returns a tuple of line objects, thus the comma

#     fig.canvas.draw()


def event_submit(text):
    global moments
    global event_number
    global moment_idx

    new_event_number = int(text)
    if event_number == new_event_number:
        return
    event_number = new_event_number
    show_img()


def moment_submit(text):
    global moments
    global event_number
    global moment_idx

    new_moment_idx = int(text)
    if moment_idx == new_moment_idx:
        return
    moment_idx = new_moment_idx
    show_img()


def show_img():
    global event_number
    global moment_idx

    img_path = os.path.join(
        '/home/hayden/cohub/events/{}/images/{}.jpg'.format(
            event_number, moment_idx))
    img = Image.open(img_path)
    img_plot.imshow(img)
    fig.canvas.draw()


event_box_axes = plt.axes([0.1, 0.15, 0.8, 0.050])
event_text_box = TextBox(event_box_axes, 'Event')
event_text_box.on_submit(event_submit)
moment_box_axes = plt.axes([0.1, 0.05, 0.8, 0.050])
moment_text_box = TextBox(moment_box_axes, 'Moment')
moment_text_box.on_submit(moment_submit)
plt.show()
