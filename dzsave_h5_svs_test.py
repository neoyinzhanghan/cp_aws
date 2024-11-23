from LLRunner.slide_processing.dzsave_h5 import dzsave_h5

svs_path = "/media/hdd3/neo/viewer_sample_huong/390359.svs"
h5_path = "/media/hdd3/neo/S3_tmp_dir/test_slide_2.h5"

dzsave_h5(svs_path, h5_path, tile_size=512, num_cpus=32, region_cropping_batch_size=256)
