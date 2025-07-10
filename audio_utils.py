import librosa
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

LABELS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
def plot_spec_heatmap(heatmap, spec, sr, pred, true, cmap='shap', include_negative=True):
  """
  Function to plot the SHAP values for one spectrogram and one prediction label.
  :param heatmap: array of heatmap to plot
  :param spec: original spectrogram of audio file
  :param sr: sampling rate
  :param pred: prediction
  :param true: ground truth 
  :param cmap: colormap to use for the heatmap
  :param include_negative: whether or not negative values are in the heatmap. If false, sets
  vmin=0 and vmax=abs(heatmap).max(). If true, sets 
  vmin=-abs(heatmap).max() and vmax=abs(heatmap).max().
  """

  figsize = [9, 2.5]

  pred_idx = LABELS[pred]

  # Define colors for the SHAP-like colormap
  colors = []
  for j in np.linspace(1, 0, 100):
    colors.append((30./255, 136./255, 229./255,j))
  for j in np.linspace(0, 1, 100):
    colors.append((255./255, 13./255, 87./255,j))

  # Create the colormap
  if cmap == 'shap':
    shap_cmap = LinearSegmentedColormap.from_list("red_transparent_blue", colors)
    cmap = shap_cmap

  # Setup figure
  fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)

  true = LABELS[true]
  pred = LABELS[pred]

  #plot the spectrogram image
  spec = np.squeeze(spec)
  librosa.display.specshow(spec, sr=sr, x_axis='time', y_axis='mel', cmap = 'viridis', ax=axes[0])
  axes[0].set_title(f'Label: {true}')

  # plot the heatmap values
  # heatmap = np.mean(heatmap, axis=-1) # aggregate the channels
  lim = abs(heatmap).max()  # get the maximum absolute value for the heatmap

  if include_negative:
    vmin = -lim
    vmax = lim
  else:
    vmin = 0
    vmax = lim

  librosa.display.specshow(spec, sr=sr, x_axis='time', y_axis='mel', cmap = 'Greys_r', ax=axes[1])
  im = librosa.display.specshow(heatmap, sr=sr, x_axis='time', y_axis='mel', cmap=cmap, ax=axes[1], vmin=vmin, vmax=vmax, alpha=0.75)
  axes[1].set_yticks([])
  axes[1].set_ylabel('')
  axes[1].set_title(f'Pred: {pred}')

  # setup colorbar
  width = 0.725
  left = (1 - width) / 2 + .01
  cax = fig.add_axes([left, -0.1, width, 0.03])  # [left, bottom, width, height]
  cb = fig.colorbar(im, cax=cax, label="Importance", orientation="horizontal",
                    aspect=figsize[0] / 0.2)
  cb.outline.set_visible(False)