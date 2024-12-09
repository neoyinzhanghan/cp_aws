import h5py

heatmap_h5 = '/media/ssd2/neo/cp_aws_playground/23.CFNA.113 A1 H&E _171848.h5'

with h5py.File(heatmap_h5, 'r') as f:
    print(f.keys())
    print(f['level_0_width'][0])
    print(f['level_0_height'][0])

    print(f[0][0, 0])