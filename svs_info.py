import openslide

svs_path = "/media/hdd3/neo/viewer_sample_huong/390359_dyadic.svs"
# svs_path = "/media/hdd3/neo/viewer_test_slide.ndpi"

wsi = openslide.OpenSlide(svs_path)

# print all the levels of the slide
print(f"Level count: {wsi.level_count}")

# get the mpp of the slide
mpp_x = float(wsi.properties[openslide.PROPERTY_NAME_MPP_X])
mpp_y = float(wsi.properties[openslide.PROPERTY_NAME_MPP_Y])

print(f"mpp_x: {mpp_x}, mpp_y: {mpp_y}")

# get the level 0 dimensions
level_0_dimensions = wsi.level_dimensions[0]

width, height = level_0_dimensions

for level in range(wsi.level_count):
    level_dimensions = wsi.level_dimensions[level]
    level_downsamples = wsi.level_downsamples[level]
    print(
        f"Level {level} dimensions: {level_dimensions}, downsample: {level_downsamples}"
    )
