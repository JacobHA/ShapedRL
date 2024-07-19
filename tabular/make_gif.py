import imageio
import matplotlib.pyplot as plt

def make_gif(images, gif_name):
    with imageio.get_writer(gif_name, mode='I', fps=5) as writer:
        for image in images:
            writer.append_data(image)

# Make a gif of the potential functions
images = []
for i in range(0, 10000, 50):
    i +=1
    try:
        # Read in the image and add a title:
        # image = plt.imread(f'potential_{i}.png')
        # plt.imshow(image)
        # plt.title(f'Potential function {i}')
        # plt.axis('off')
        # plt.savefig(f'potential_{i}.png')
        images.append(imageio.imread(f'potential_{i}.png'))
    except FileNotFoundError:
        continue

make_gif(images, 'potential_functions.gif')
