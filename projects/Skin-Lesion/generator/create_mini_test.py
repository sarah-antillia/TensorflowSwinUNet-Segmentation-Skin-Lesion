import os
import shutil
import glob
import random
import traceback


def create_mini_test(images_dir, masks_dir, output_images_dir, output_masks_dir, sample=20):
  if os.path.exists(output_images_dir):
    shutil.rmtree(output_images_dir)
  if not os.path.exists(output_images_dir):
    os.makedirs(output_images_dir)

  if os.path.exists(output_masks_dir):
    shutil.rmtree(output_masks_dir)
  if not os.path.exists(output_masks_dir):
    os.makedirs(output_masks_dir)

  image_files = glob.glob (images_dir + "/*.jpg")
  image_files = sorted(image_files)
  image_files = random.sample(image_files, sample)

  for image_file in image_files:
    shutil.copy2(image_file, output_images_dir)
    print("Copied {}".format(image_file))

    basename = os.path.basename(image_file).split(".")[0]
    mask_file = basename + "_segmentation.png"

    mask_filepath = os.path.join(masks_dir, mask_file)
    shutil.copy2(mask_filepath, output_masks_dir)
    print("Copied {}".format(mask_filepath))



if __name__ == "__main__":
  try:
    images_dir        = "./ISIC-2017_Test_v2_Data"
    masks_dir         = "./ISIC-2017_Test_v2_Part1_GroundTruth"
    output_images_dir = "./mini_test/images/"
    output_masks_dir  = "./mini_test/masks/"
    create_mini_test(images_dir, masks_dir, output_images_dir, output_masks_dir, sample=20)

  except:
    traceback.print_exc()
