import numpy as np

def save_array(filename, scores):
    with open(filename, "w+") as f:
        np.savetxt(f, scores, delimiter=",")

def load_array(filename):
    with open(filename) as f:
        array = np.loadtxt(f, delimiter=",")
    return array

def nuke_screenshots():
    extension = (".png", ".gif")
    for file in folder.iterdir():
        if file.suffix.lower() in extension:
            file.unlink()
    print("Successfully removed screenshots/animations!")


def save_screenshot(subdir, file_name, image):
    plt.imsave(f"./datasets/pacman_screenshots/{subdir}/{file_name}.png", image)


def build_animation(subdir, modifier_like, animation_name):
    screenshots_in_dir = sorted(folder.glob(modifier_like))
    frames = []
    for image in screenshots_in_dir:
        img = imageio.v2.imread(f"./datasets/pacman_screenshots/{subdir}/{image.name}")
        frames.append(img)
    imageio.v2.mimsave(f"./datasets/pacman_screenshots/{subdir}/{animation_name}.gif", frames, fps=15)


