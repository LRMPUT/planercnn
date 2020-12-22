import numpy as np
import os
import open3d as o3d


def main():
    print('reading point cloud')
    pointcloud = o3d.io.read_point_cloud('/mnt/data/TERRINet/2019_10_10/scene1/cloud.pcd')
    o3d.visualization.draw_geometries([pointcloud])
    print('estimating normals')
    pointcloud.estimate_normals()
    print('creating mesh')
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pointcloud,
    #                                                                  depth=10,
    #                                                                  width=0,
    #                                                                  scale=1.1, linear_fit=False)[0]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pointcloud, o3d.utility.DoubleVector([0.1]))
    print('saving mesh')
    o3d.io.write_triangle_mesh('/mnt/data/TERRINet/2019_10_10/scene1/mesh.ply', mesh)
    o3d.visualization.draw_geometries([mesh])


if __name__=='__main__':
    main()
