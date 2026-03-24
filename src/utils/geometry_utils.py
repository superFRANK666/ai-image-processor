"""
3D几何计算工具类
提供旋转、投影等3D数学运算的工具函数
"""
import numpy as np
import math
from typing import Tuple


class GeometryUtils:
    """
    3D几何工具类
    集中管理3D数学计算逻辑，便于测试和维护
    """
    
    @staticmethod
    def get_rotation_matrix(angle: float, axis: str = 'y') -> np.ndarray:
        """
        获取旋转矩阵
        
        Args:
            angle: 旋转角度（弧度）
            axis: 旋转轴 ('x', 'y', 'z')
        
        Returns:
            3x3 旋转矩阵
        """
        c, s = math.cos(angle), math.sin(angle)
        
        if axis == 'x':
            return np.array([
                [1, 0, 0],
                [0, c, -s],
                [0, s, c]
            ], dtype=np.float32)
        elif axis == 'y':
            return np.array([
                [c, 0, s],
                [0, 1, 0],
                [-s, 0, c]
            ], dtype=np.float32)
        elif axis == 'z':
            return np.array([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]
            ], dtype=np.float32)
        else:
            raise ValueError(f"不支持的旋转轴: {axis}, 应为 'x', 'y' 或 'z'")
    
    @staticmethod
    def rotate_points(points: np.ndarray, angle: float, axis: str = 'y') -> np.ndarray:
        """
        旋转3D点集
        
        Args:
            points: 点坐标数组 (N, 3)
            angle: 旋转角度（弧度）
            axis: 旋转轴
        
        Returns:
            旋转后的点坐标数组 (N, 3)
        """
        R = GeometryUtils.get_rotation_matrix(angle, axis)
        return points @ R.T
    
    @staticmethod
    def orthographic_projection(points: np.ndarray, 
                                width: int, 
                                height: int,
                                scale: float = 0.4) -> Tuple[np.ndarray, np.ndarray]:
        """
        正交投影
        将3D点投影到2D屏幕坐标
        
        Args:
            points: 3D点坐标 (N, 3)
            width: 屏幕宽度
            height: 屏幕高度
            scale: 缩放因子
        
        Returns:
            (projected_2d, z_values): 2D投影坐标 (N, 2) 和 Z值 (N,)
        """
        # 计算缩放和中心
        scale_pixels = min(width, height) * scale
        cx, cy = width // 2, height // 2
        
        # 投影到2D（注意Y轴翻转，因为屏幕坐标系Y向下）
        x_2d = points[:, 0] * scale_pixels + cx
        y_2d = -points[:, 1] * scale_pixels + cy  # Y轴翻转
        
        projected_2d = np.stack([x_2d, y_2d], axis=-1)
        z_values = points[:, 2]
        
        return projected_2d, z_values
    
    @staticmethod
    def compute_face_normal(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """
        计算三角面的法线
        
        Args:
            v0, v1, v2: 三角形的三个顶点 (3,)
        
        Returns:
            单位法线向量 (3,)
        """
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        
        # 归一化
        norm = np.linalg.norm(normal)
        if norm > 1e-6:
            normal = normal / norm
        
        return normal
    
    @staticmethod
    def compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """
        计算顶点法线（通过相邻面法线平均）
        
        Args:
            vertices: 顶点坐标 (N, 3)
            faces: 面索引 (M, 3)
        
        Returns:
            顶点法线 (N, 3)
        """
        normals = np.zeros_like(vertices)
        
        for face in faces:
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            edge1 = v1 - v0
            edge2 = v2 - v0
            face_normal = np.cross(edge1, edge2)
            
            # 累加到各顶点
            normals[face[0]] += face_normal
            normals[face[1]] += face_normal
            normals[face[2]] += face_normal
        
        # 归一化
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / (norms + 1e-6)
        
        return normals.astype(np.float32)
    
    @staticmethod
    def simple_lighting(face_normal: np.ndarray, 
                       light_direction: np.ndarray = None) -> float:
        """
        简单的漫反射光照计算
        
        Args:
            face_normal: 面法线
            light_direction: 光源方向（默认从右上前方）
        
        Returns:
            光照强度 [0, 1]
        """
        if light_direction is None:
            # 默认光源从右上前方照射
            light_direction = np.array([0.5, 0.5, 1.0])
            light_direction = light_direction / np.linalg.norm(light_direction)
        
        # Lambert漫反射
        intensity = np.dot(face_normal, light_direction)
        # 限制在 [0, 1] 范围，环境光0.3
        intensity = max(0.0, intensity) * 0.7 + 0.3
        
        return min(1.0, intensity)
    
    @staticmethod
    def depth_based_lighting(z_value: float, z_min: float, z_max: float) -> float:
        """
        基于深度的简单光照
        
        Args:
            z_value: 当前点的Z值
            z_min: 最小Z值
            z_max: 最大Z值
        
        Returns:
            光照系数 [0.5, 1.0]
        """
        if z_max - z_min < 1e-6:
            return 1.0
        
        # 归一化深度 [0, 1]
        normalized_depth = (z_value - z_min) / (z_max - z_min)
        
        # 接近相机（深度大）时更亮
        light_factor = 0.5 + 0.5 * normalized_depth
        
        return light_factor
    
    @staticmethod
    def is_point_in_screen(x: float, y: float, width: int, height: int) -> bool:
        """
        检查点是否在屏幕范围内
        
        Args:
            x, y: 点坐标
            width: 屏幕宽度
            height: 屏幕高度
        
        Returns:
            是否在屏幕内
        """
        return 0 <= x < width and 0 <= y < height
    
    @staticmethod
    def sort_faces_by_depth(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
        """
        按深度排序面（画家算法）
        
        Args:
            vertices: 顶点坐标 (N, 3)
            faces: 面索引 (M, 3)
        
        Returns:
            排序后的面索引数组
        """
        # 计算每个面的平均深度
        face_depths = np.mean(vertices[faces, 2], axis=1)
        
        # 从远到近排序
        sorted_indices = np.argsort(face_depths)
        
        return sorted_indices
    
    @staticmethod
    def create_grid_mesh(width: int, height: int) -> np.ndarray:
        """
        创建网格面索引
        
        Args:
            width: 网格宽度（列数）
            height: 网格高度（行数）
        
        Returns:
            面索引数组 ((width-1)*(height-1)*2, 3)
        """
        faces = []
        
        for i in range(height - 1):
            for j in range(width - 1):
                # 顶点索引
                v0 = i * width + j
                v1 = i * width + (j + 1)
                v2 = (i + 1) * width + j
                v3 = (i + 1) * width + (j + 1)
                
                # 两个三角形组成一个四边形
                faces.append([v0, v2, v1])
                faces.append([v1, v2, v3])
        
        return np.array(faces, dtype=np.int32)
    
    @staticmethod
    def create_uv_coordinates(width: int, height: int) -> np.ndarray:
        """
        创建UV纹理坐标
        
        Args:
            width: 网格宽度
            height: 网格高度
        
        Returns:
            UV坐标 (width*height, 2)
        """
        u = np.linspace(0, 1, width)
        v = np.linspace(0, 1, height)
        uu, vv = np.meshgrid(u, v)
        uvs = np.stack([uu, vv], axis=-1).reshape(-1, 2)
        
        return uvs.astype(np.float32)
