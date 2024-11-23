import openslide

svs_path = "/media/hdd3/neo/viewer_sample_huong/390359.svs"

wsi = openslide.OpenSlide(svs_path)

# print all the levels of the slide
print(wsi.level_count)

# get the mpp of the slide
mpp_x = float(wsi.properties[openslide.PROPERTY_NAME_MPP_X])

