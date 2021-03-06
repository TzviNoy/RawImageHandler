import matplotlib.pyplot as plt


# Display preprocessing:

for count, element in enumerate(formatted):
    print(f"int16 representation: {int(element, 2)}\n"
          f"16 bits binary representation: {element}\n"
          f"Hexadecimal representation: {hex(int(element, 2))}\n"
          f"8 bits hexadecimal representation: {hex(int(element[:8], 2))}\n")
    if count == 5:
        break

# Display histograms

images = [clipped_image, normalized_image, equalized_image, adapt_equalized_image]

fig = plt.figure()
gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)
axs = gs.subplots(sharex='col', sharey='row')
fig.suptitle('Sharing x per column, y per row')

for counter, image in enumerate(images):
    hist, bins = np.histogram(image, 2**8)
    axs.ravel()[counter] = plt.subplot(2, 2, counter + 1)
    axs.ravel()[counter].plot(bins[:-1], hist)
plt.tight_layout()
plt.show()


# Display processed image
images_figure, axs = plt.subplots(2, 2)
axs[0, 0].imshow(clipped_image, cmap='gray', vmin=clipped_image.min(), vmax=clipped_image.max())
axs[0, 1].imshow(normalized_image, cmap='gray')
axs[1, 0].imshow(equalized_image, cmap='gray')
axs[1, 1].imshow(adapt_equalized_image, cmap='gray')
plt.show()