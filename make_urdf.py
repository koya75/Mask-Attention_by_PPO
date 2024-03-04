# 3D item を読み込むためのurdf ファイルを作成します
import os
import argparse
from pathlib import Path

class URDF:
    def urdf01(self,obj_path: Path, urdf_dir: Path):
        xml = """<?xml version="1.0"?>
    <robot name="{name}">
    <link name="base_link">
        <visual>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0025 0.0025 0.0025"/>
        </geometry>
        <material>
            <color rgba="1 1 1 1"/>
        </material>
        </visual>
        <collision>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0025 0.0025 0.0025"/>
        </geometry>
        </collision>
    </link>
    </robot>
    """.format(
            name=obj_path.stem, path=obj_path.resolve()
        )
        return xml

    def urdf02(self,obj_path: Path, urdf_dir: Path):
        xml = """<?xml version="1.0"?>
    <robot name="{name}">
    <link name="base_link">
        <visual>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0025 0.0025 0.0025"/>
        </geometry>
        <material>
            <color rgba="1 1 1 1"/>
        </material>
        </visual>
        <collision>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0025 0.0025 0.0025"/>
        </geometry>
        </collision>
    </link>
    </robot>
    """.format(
            name=obj_path.stem, path=obj_path.resolve()
        )
        return xml

    def urdf03(self,obj_path: Path, urdf_dir: Path):
        xml = """<?xml version="1.0"?>
    <robot name="{name}">
    <link name="base_link">
        <visual>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0028 0.0028 0.0028"/>
        </geometry>
        <material>
            <color rgba="1 1 1 1"/>
        </material>
        </visual>
        <collision>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0028 0.0028 0.0028"/>
        </geometry>
        </collision>
    </link>
    </robot>
    """.format(
            name=obj_path.stem, path=obj_path.resolve()
        )
        return xml

    def urdf04(self,obj_path: Path, urdf_dir: Path):
        xml = """<?xml version="1.0"?>
    <robot name="{name}">
    <link name="base_link">
        <visual>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0028 0.0028 0.0028"/>
        </geometry>
        <material>
            <color rgba="1 1 1 1"/>
        </material>
        </visual>
        <collision>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0028 0.0028 0.0028"/>
        </geometry>
        </collision>
    </link>
    </robot>
    """.format(
            name=obj_path.stem, path=obj_path.resolve()
        )
        return xml

    def urdf05(self,obj_path: Path, urdf_dir: Path):
        xml = """<?xml version="1.0"?>
    <robot name="{name}">
    <link name="base_link">
        <visual>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0025 0.0025 0.0025"/>
        </geometry>
        <material>
            <color rgba="1 1 1 1"/>
        </material>
        </visual>
        <collision>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0025 0.0025 0.0025"/>
        </geometry>
        </collision>
    </link>
    </robot>
    """.format(
            name=obj_path.stem, path=obj_path.resolve()
        )
        return xml

    def urdf06(self,obj_path: Path, urdf_dir: Path):
        xml = """<?xml version="1.0"?>
    <robot name="{name}">
    <link name="base_link">
        <visual>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.001 0.001 0.001"/>
        </geometry>
        <material>
            <color rgba="1 1 1 1"/>
        </material>
        </visual>
        <collision>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.001 0.001 0.001"/>
        </geometry>
        </collision>
    </link>
    </robot>
    """.format(
            name=obj_path.stem, path=obj_path.resolve()
        )
        return xml

    def urdf07(self,obj_path: Path, urdf_dir: Path):
        xml = """<?xml version="1.0"?>
    <robot name="{name}">
    <link name="base_link">
        <visual>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0025 0.0025 0.0025"/>
        </geometry>
        <material>
            <color rgba="1 1 1 1"/>
        </material>
        </visual>
        <collision>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0025 0.0025 0.0025"/>
        </geometry>
        </collision>
    </link>
    </robot>
    """.format(
            name=obj_path.stem, path=obj_path.resolve()
        )
        return xml

    def urdf08(self,obj_path: Path, urdf_dir: Path):
        xml = """<?xml version="1.0"?>
    <robot name="{name}">
    <link name="base_link">
        <visual>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0025 0.0025 0.0025"/>
        </geometry>
        <material>
            <color rgba="1 1 1 1"/>
        </material>
        </visual>
        <collision>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0025 0.0025 0.0025"/>
        </geometry>
        </collision>
    </link>
    </robot>
    """.format(
            name=obj_path.stem, path=obj_path.resolve()
        )
        return xml

    def urdf09(self,obj_path: Path, urdf_dir: Path):
        xml = """<?xml version="1.0"?>
    <robot name="{name}">
    <link name="base_link">
        <visual>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0023 0.0023 0.0023"/>
        </geometry>
        <material>
            <color rgba="1 1 1 1"/>
        </material>
        </visual>
        <collision>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0023 0.0023 0.0023"/>
        </geometry>
        </collision>
    </link>
    </robot>
    """.format(
            name=obj_path.stem, path=obj_path.resolve()
        )
        return xml

    def urdf10(self,obj_path: Path, urdf_dir: Path):
        xml = """<?xml version="1.0"?>
    <robot name="{name}">
    <link name="base_link">
        <visual>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.001 0.001 0.001"/>
        </geometry>
        <material>
            <color rgba="1 1 1 1"/>
        </material>
        </visual>
        <collision>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.001 0.001 0.001"/>
        </geometry>
        </collision>
    </link>
    </robot>
    """.format(
            name=obj_path.stem, path=obj_path.resolve()
        )
        return xml

    def urdf11(self,obj_path: Path, urdf_dir: Path):
        xml = """<?xml version="1.0"?>
    <robot name="{name}">
    <link name="base_link">
        <visual>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.001 0.001 0.001"/>
        </geometry>
        <material>
            <color rgba="1 1 1 1"/>
        </material>
        </visual>
        <collision>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.001 0.001 0.001"/>
        </geometry>
        </collision>
    </link>
    </robot>
    """.format(
            name=obj_path.stem, path=obj_path.resolve()
        )
        return xml

    def urdf12(self,obj_path: Path, urdf_dir: Path):
        xml = """<?xml version="1.0"?>
    <robot name="{name}">
    <link name="base_link">
        <visual>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0025 0.0025 0.0025"/>
        </geometry>
        <material>
            <color rgba="1 1 1 1"/>
        </material>
        </visual>
        <collision>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0025 0.0025 0.0025"/>
        </geometry>
        </collision>
    </link>
    </robot>
    """.format(
            name=obj_path.stem, path=obj_path.resolve()
        )
        return xml

    def urdf13(self,obj_path: Path, urdf_dir: Path):
        xml = """<?xml version="1.0"?>
    <robot name="{name}">
    <link name="base_link">
        <visual>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0025 0.0025 0.0025"/>
        </geometry>
        <material>
            <color rgba="1 1 1 1"/>
        </material>
        </visual>
        <collision>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0025 0.0025 0.0025"/>
        </geometry>
        </collision>
    </link>
    </robot>
    """.format(
            name=obj_path.stem, path=obj_path.resolve()
        )
        return xml

    def urdf14(self,obj_path: Path, urdf_dir: Path):
        xml = """<?xml version="1.0"?>
    <robot name="{name}">
    <link name="base_link">
        <visual>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0025 0.0025 0.0025"/>
        </geometry>
        <material>
            <color rgba="1 1 1 1"/>
        </material>
        </visual>
        <collision>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0025 0.0025 0.0025"/>
        </geometry>
        </collision>
    </link>
    </robot>
    """.format(
            name=obj_path.stem, path=obj_path.resolve()
        )
        return xml

    def urdf15(self,obj_path: Path, urdf_dir: Path):
        xml = """<?xml version="1.0"?>
    <robot name="{name}">
    <link name="base_link">
        <visual>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.001 0.001 0.001"/>
        </geometry>
        <material>
            <color rgba="1 1 1 1"/>
        </material>
        </visual>
        <collision>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.001 0.001 0.001"/>
        </geometry>
        </collision>
    </link>
    </robot>
    """.format(
            name=obj_path.stem, path=obj_path.resolve()
        )
        return xml

    def urdf16(self,obj_path: Path, urdf_dir: Path):
        xml = """<?xml version="1.0"?>
    <robot name="{name}">
    <link name="base_link">
        <visual>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0025 0.0025 0.0025"/>
        </geometry>
        <material>
            <color rgba="1 1 1 1"/>
        </material>
        </visual>
        <collision>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0025 0.0025 0.0025"/>
        </geometry>
        </collision>
    </link>
    </robot>
    """.format(
            name=obj_path.stem, path=obj_path.resolve()
        )
        return xml

    def urdf17(self,obj_path: Path, urdf_dir: Path):
        xml = """<?xml version="1.0"?>
    <robot name="{name}">
    <link name="base_link">
        <visual>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0025 0.0025 0.0025"/>
        </geometry>
        <material>
            <color rgba="1 1 1 1"/>
        </material>
        </visual>
        <collision>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0025 0.0025 0.0025"/>
        </geometry>
        </collision>
    </link>
    </robot>
    """.format(
            name=obj_path.stem, path=obj_path.resolve()
        )
        return xml

    def urdf18(self,obj_path: Path, urdf_dir: Path):
        xml = """<?xml version="1.0"?>
    <robot name="{name}">
    <link name="base_link">
        <visual>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0025 0.0025 0.0025"/>
        </geometry>
        <material>
            <color rgba="1 1 1 1"/>
        </material>
        </visual>
        <collision>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0025 0.0025 0.0025"/>
        </geometry>
        </collision>
    </link>
    </robot>
    """.format(
            name=obj_path.stem, path=obj_path.resolve()
        )
        return xml

    def urdf19(self,obj_path: Path, urdf_dir: Path):
        xml = """<?xml version="1.0"?>
    <robot name="{name}">
    <link name="base_link">
        <visual>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0025 0.0025 0.0025"/>
        </geometry>
        <material>
            <color rgba="1 1 1 1"/>
        </material>
        </visual>
        <collision>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0025 0.0025 0.0025"/>
        </geometry>
        </collision>
    </link>
    </robot>
    """.format(
            name=obj_path.stem, path=obj_path.resolve()
        )
        return xml

    def urdf20(self,obj_path: Path, urdf_dir: Path):
        xml = """<?xml version="1.0"?>
    <robot name="{name}">
    <link name="base_link">
        <visual>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0025 0.0025 0.0025"/>
        </geometry>
        <material>
            <color rgba="1 1 1 1"/>
        </material>
        </visual>
        <collision>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0025 0.0025 0.0025"/>
        </geometry>
        </collision>
    </link>
    </robot>
    """.format(
            name=obj_path.stem, path=obj_path.resolve()
        )
        return xml

    def urdf21(self,obj_path: Path, urdf_dir: Path):
        xml = """<?xml version="1.0"?>
    <robot name="{name}">
    <link name="base_link">
        <visual>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.001 0.001 0.001"/>
        </geometry>
        <material>
            <color rgba="1 1 1 1"/>
        </material>
        </visual>
        <collision>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.001 0.001 0.001"/>
        </geometry>
        </collision>
    </link>
    </robot>
    """.format(
            name=obj_path.stem, path=obj_path.resolve()
        )
        return xml

    def urdf22(self,obj_path: Path, urdf_dir: Path):
        xml = """<?xml version="1.0"?>
    <robot name="{name}">
    <link name="base_link">
        <visual>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0025 0.0025 0.0025"/>
        </geometry>
        <material>
            <color rgba="1 1 1 1"/>
        </material>
        </visual>
        <collision>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0025 0.0025 0.0025"/>
        </geometry>
        </collision>
    </link>
    </robot>
    """.format(
            name=obj_path.stem, path=obj_path.resolve()
        )
        return xml

    def urdf23(self,obj_path: Path, urdf_dir: Path):
        xml = """<?xml version="1.0"?>
    <robot name="{name}">
    <link name="base_link">
        <visual>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0023 0.0023 0.0023"/>
        </geometry>
        <material>
            <color rgba="1 1 1 1"/>
        </material>
        </visual>
        <collision>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0023 0.0023 0.0023"/>
        </geometry>
        </collision>
    </link>
    </robot>
    """.format(
            name=obj_path.stem, path=obj_path.resolve()
        )
        return xml

    def urdf24(self,obj_path: Path, urdf_dir: Path):
        xml = """<?xml version="1.0"?>
    <robot name="{name}">
    <link name="base_link">
        <visual>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0025 0.0025 0.0025"/>
        </geometry>
        <material>
            <color rgba="1 1 1 1"/>
        </material>
        </visual>
        <collision>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0025 0.0025 0.0025"/>
        </geometry>
        </collision>
    </link>
    </robot>
    """.format(
            name=obj_path.stem, path=obj_path.resolve()
        )
        return xml

    def urdf25(self,obj_path: Path, urdf_dir: Path):
        xml = """<?xml version="1.0"?>
    <robot name="{name}">
    <link name="base_link">
        <visual>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0025 0.0025 0.0025"/>
        </geometry>
        <material>
            <color rgba="1 1 1 1"/>
        </material>
        </visual>
        <collision>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0025 0.0025 0.0025"/>
        </geometry>
        </collision>
    </link>
    </robot>
    """.format(
            name=obj_path.stem, path=obj_path.resolve()
        )
        return xml

    def urdf26(self,obj_path: Path, urdf_dir: Path):
        xml = """<?xml version="1.0"?>
    <robot name="{name}">
    <link name="base_link">
        <visual>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0025 0.0025 0.0025"/>
        </geometry>
        <material>
            <color rgba="1 1 1 1"/>
        </material>
        </visual>
        <collision>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0025 0.0025 0.0025"/>
        </geometry>
        </collision>
    </link>
    </robot>
    """.format(
            name=obj_path.stem, path=obj_path.resolve()
        )
        return xml

    def urdf27(self,obj_path: Path, urdf_dir: Path):
        xml = """<?xml version="1.0"?>
    <robot name="{name}">
    <link name="base_link">
        <visual>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0025 0.0025 0.0025"/>
        </geometry>
        <material>
            <color rgba="1 1 1 1"/>
        </material>
        </visual>
        <collision>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0025 0.0025 0.0025"/>
        </geometry>
        </collision>
    </link>
    </robot>
    """.format(
            name=obj_path.stem, path=obj_path.resolve()
        )
        return xml

    def urdf28(self,obj_path: Path, urdf_dir: Path):
        xml = """<?xml version="1.0"?>
    <robot name="{name}">
    <link name="base_link">
        <visual>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0025 0.0025 0.0025"/>
        </geometry>
        <material>
            <color rgba="1 1 1 1"/>
        </material>
        </visual>
        <collision>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0025 0.0025 0.0025"/>
        </geometry>
        </collision>
    </link>
    </robot>
    """.format(
            name=obj_path.stem, path=obj_path.resolve()
        )
        return xml

    def urdf29(self,obj_path: Path, urdf_dir: Path):
        xml = """<?xml version="1.0"?>
    <robot name="{name}">
    <link name="base_link">
        <visual>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.001 0.001 0.001"/>
        </geometry>
        <material>
            <color rgba="1 1 1 1"/>
        </material>
        </visual>
        <collision>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.001 0.001 0.001"/>
        </geometry>
        </collision>
    </link>
    </robot>
    """.format(
            name=obj_path.stem, path=obj_path.resolve()
        )
        return xml

    def urdf30(self,obj_path: Path, urdf_dir: Path):
        xml = """<?xml version="1.0"?>
    <robot name="{name}">
    <link name="base_link">
        <visual>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0025 0.0025 0.0025"/>
        </geometry>
        <material>
            <color rgba="1 1 1 1"/>
        </material>
        </visual>
        <collision>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0025 0.0025 0.0025"/>
        </geometry>
        </collision>
    </link>
    </robot>
    """.format(
            name=obj_path.stem, path=obj_path.resolve()
        )
        return xml

    def urdf31(self,obj_path: Path, urdf_dir: Path):
        xml = """<?xml version="1.0"?>
    <robot name="{name}">
    <link name="base_link">
        <visual>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0025 0.0025 0.0025"/>
        </geometry>
        <material>
            <color rgba="1 1 1 1"/>
        </material>
        </visual>
        <collision>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0025 0.0025 0.0025"/>
        </geometry>
        </collision>
    </link>
    </robot>
    """.format(
            name=obj_path.stem, path=obj_path.resolve()
        )
        return xml

    def urdf32(self,obj_path: Path, urdf_dir: Path):
        xml = """<?xml version="1.0"?>
    <robot name="{name}">
    <link name="base_link">
        <visual>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0025 0.0025 0.0025"/>
        </geometry>
        <material>
            <color rgba="1 1 1 1"/>
        </material>
        </visual>
        <collision>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0025 0.0025 0.0025"/>
        </geometry>
        </collision>
    </link>
    </robot>
    """.format(
            name=obj_path.stem, path=obj_path.resolve()
        )
        return xml

    def urdf33(self,obj_path: Path, urdf_dir: Path):
        xml = """<?xml version="1.0"?>
    <robot name="{name}">
    <link name="base_link">
        <visual>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.001 0.001 0.001"/>
        </geometry>
        <material>
            <color rgba="1 1 1 1"/>
        </material>
        </visual>
        <collision>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.001 0.001 0.001"/>
        </geometry>
        </collision>
    </link>
    </robot>
    """.format(
            name=obj_path.stem, path=obj_path.resolve()
        )
        return xml

    def urdf34(self,obj_path: Path, urdf_dir: Path):
        xml = """<?xml version="1.0"?>
    <robot name="{name}">
    <link name="base_link">
        <visual>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0025 0.0025 0.0025"/>
        </geometry>
        <material>
            <color rgba="1 1 1 1"/>
        </material>
        </visual>
        <collision>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0025 0.0025 0.0025"/>
        </geometry>
        </collision>
    </link>
    </robot>
    """.format(
            name=obj_path.stem, path=obj_path.resolve()
        )
        return xml

    def urdf35(self,obj_path: Path, urdf_dir: Path):
        xml = """<?xml version="1.0"?>
    <robot name="{name}">
    <link name="base_link">
        <visual>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.003 0.003 0.003"/>
        </geometry>
        <material>
            <color rgba="1 1 1 1"/>
        </material>
        </visual>
        <collision>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.003 0.003 0.003"/>
        </geometry>
        </collision>
    </link>
    </robot>
    """.format(
            name=obj_path.stem, path=obj_path.resolve()
        )
        return xml

    def urdf36(self,obj_path: Path, urdf_dir: Path):
        xml = """<?xml version="1.0"?>
    <robot name="{name}">
    <link name="base_link">
        <visual>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0028 0.0028 0.0028"/>
        </geometry>
        <material>
            <color rgba="1 1 1 1"/>
        </material>
        </visual>
        <collision>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0028 0.0028 0.0028"/>
        </geometry>
        </collision>
    </link>
    </robot>
    """.format(
            name=obj_path.stem, path=obj_path.resolve()
        )
        return xml

    def urdf37(self,obj_path: Path, urdf_dir: Path):
        xml = """<?xml version="1.0"?>
    <robot name="{name}">
    <link name="base_link">
        <visual>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0025 0.0025 0.0025"/>
        </geometry>
        <material>
            <color rgba="1 1 1 1"/>
        </material>
        </visual>
        <collision>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0025 0.0025 0.0025"/>
        </geometry>
        </collision>
    </link>
    </robot>
    """.format(
            name=obj_path.stem, path=obj_path.resolve()
        )
        return xml

    def urdf38(self,obj_path: Path, urdf_dir: Path):
        xml = """<?xml version="1.0"?>
    <robot name="{name}">
    <link name="base_link">
        <visual>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0025 0.0025 0.0025"/>
        </geometry>
        <material>
            <color rgba="1 1 1 1"/>
        </material>
        </visual>
        <collision>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.0025 0.0025 0.0025"/>
        </geometry>
        </collision>
    </link>
    </robot>
    """.format(
            name=obj_path.stem, path=obj_path.resolve()
        )
        return xml

    def urdf39(self,obj_path: Path, urdf_dir: Path):
        xml = """<?xml version="1.0"?>
    <robot name="{name}">
    <link name="base_link">
        <visual>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.002 0.002 0.002"/>
        </geometry>
        <material>
            <color rgba="1 1 1 1"/>
        </material>
        </visual>
        <collision>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.002 0.002 0.002"/>
        </geometry>
        </collision>
    </link>
    </robot>
    """.format(
            name=obj_path.stem, path=obj_path.resolve()
        )
        return xml

    def urdf40_1(self,obj_path: Path, urdf_dir: Path):
        xml = """<?xml version="1.0"?>
    <robot name="{name}">
    <link name="base_link">
        <visual>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.001 0.001 0.001"/>
        </geometry>
        <material>
            <color rgba="1 1 1 1"/>
        </material>
        </visual>
        <collision>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.001 0.001 0.001"/>
        </geometry>
        </collision>
    </link>
    </robot>
    """.format(
            name=obj_path.stem, path=obj_path.resolve()
        )
        return xml

    def urdf40_2(self,obj_path: Path, urdf_dir: Path):
        xml = """<?xml version="1.0"?>
    <robot name="{name}">
    <link name="base_link">
        <visual>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.001 0.001 0.001"/>
        </geometry>
        <material>
            <color rgba="1 1 1 1"/>
        </material>
        </visual>
        <collision>
        <origin xyz="0 0 0"/>
        <geometry>
            <mesh filename="{path}" scale="0.001 0.001 0.001"/>
        </geometry>
        </collision>
    </link>
    </robot>
    """.format(
            name=obj_path.stem, path=obj_path.resolve()
        )
        return xml


    def create_urdf(self):
        ARCdataset_dir: Path = Path("3Dmodels")
        out_urdf_dir: Path = Path("./urdf")
        os.makedirs(out_urdf_dir, exist_ok=True)

        items = sorted(list(ARCdataset_dir.glob("*/item*.obj")))
        for item in items:
            item_urdf = item.stem + ".urdf"
            urdf_path = out_urdf_dir.joinpath(item_urdf)
            if str(item.stem) == "item01":
                with open(urdf_path, mode="w", encoding="UTF-8") as f:
                    f.write(self.urdf01(item, out_urdf_dir))
            elif str(item.stem) == "item02":
                with open(urdf_path, mode="w", encoding="UTF-8") as f:
                    f.write(self.urdf02(item, out_urdf_dir))
            elif str(item.stem) == "item03":
                with open(urdf_path, mode="w", encoding="UTF-8") as f:
                    f.write(self.urdf03(item, out_urdf_dir))
            elif str(item.stem) == "item04":
                with open(urdf_path, mode="w", encoding="UTF-8") as f:
                    f.write(self.urdf04(item, out_urdf_dir))
            elif str(item.stem) == "item05":
                with open(urdf_path, mode="w", encoding="UTF-8") as f:
                    f.write(self.urdf05(item, out_urdf_dir))
            elif str(item.stem) == "item06":
                with open(urdf_path, mode="w", encoding="UTF-8") as f:
                    f.write(self.urdf06(item, out_urdf_dir))
            elif str(item.stem) == "item07":
                with open(urdf_path, mode="w", encoding="UTF-8") as f:
                    f.write(self.urdf07(item, out_urdf_dir))
            elif str(item.stem) == "item08":
                with open(urdf_path, mode="w", encoding="UTF-8") as f:
                    f.write(self.urdf08(item, out_urdf_dir))
            elif str(item.stem) == "item09":
                with open(urdf_path, mode="w", encoding="UTF-8") as f:
                    f.write(self.urdf09(item, out_urdf_dir))
            elif str(item.stem) == "item10":
                with open(urdf_path, mode="w", encoding="UTF-8") as f:
                    f.write(self.urdf10(item, out_urdf_dir))
            elif str(item.stem) == "item11":
                with open(urdf_path, mode="w", encoding="UTF-8") as f:
                    f.write(self.urdf11(item, out_urdf_dir))
            elif str(item.stem) == "item12":
                with open(urdf_path, mode="w", encoding="UTF-8") as f:
                    f.write(self.urdf12(item, out_urdf_dir))
            elif str(item.stem) == "item13":
                with open(urdf_path, mode="w", encoding="UTF-8") as f:
                    f.write(self.urdf13(item, out_urdf_dir))
            elif str(item.stem) == "item14":
                with open(urdf_path, mode="w", encoding="UTF-8") as f:
                    f.write(self.urdf14(item, out_urdf_dir))
            elif str(item.stem) == "item15":
                with open(urdf_path, mode="w", encoding="UTF-8") as f:
                    f.write(self.urdf15(item, out_urdf_dir))
            elif str(item.stem) == "item16":
                with open(urdf_path, mode="w", encoding="UTF-8") as f:
                    f.write(self.urdf16(item, out_urdf_dir))
            elif str(item.stem) == "item17":
                with open(urdf_path, mode="w", encoding="UTF-8") as f:
                    f.write(self.urdf17(item, out_urdf_dir))
            elif str(item.stem) == "item18":
                with open(urdf_path, mode="w", encoding="UTF-8") as f:
                    f.write(self.urdf18(item, out_urdf_dir))
            elif str(item.stem) == "item19":
                with open(urdf_path, mode="w", encoding="UTF-8") as f:
                    f.write(self.urdf19(item, out_urdf_dir))
            elif str(item.stem) == "item20":
                with open(urdf_path, mode="w", encoding="UTF-8") as f:
                    f.write(self.urdf20(item, out_urdf_dir))
            elif str(item.stem) == "item21":
                with open(urdf_path, mode="w", encoding="UTF-8") as f:
                    f.write(self.urdf21(item, out_urdf_dir))
            elif str(item.stem) == "item22":
                with open(urdf_path, mode="w", encoding="UTF-8") as f:
                    f.write(self.urdf22(item, out_urdf_dir))
            elif str(item.stem) == "item23":
                with open(urdf_path, mode="w", encoding="UTF-8") as f:
                    f.write(self.urdf23(item, out_urdf_dir))
            elif str(item.stem) == "item24":
                with open(urdf_path, mode="w", encoding="UTF-8") as f:
                    f.write(self.urdf24(item, out_urdf_dir))
            elif str(item.stem) == "item25":
                with open(urdf_path, mode="w", encoding="UTF-8") as f:
                    f.write(self.urdf25(item, out_urdf_dir))
            elif str(item.stem) == "item26":
                with open(urdf_path, mode="w", encoding="UTF-8") as f:
                    f.write(self.urdf26(item, out_urdf_dir))
            elif str(item.stem) == "item27":
                with open(urdf_path, mode="w", encoding="UTF-8") as f:
                    f.write(self.urdf27(item, out_urdf_dir))
            elif str(item.stem) == "item28":
                with open(urdf_path, mode="w", encoding="UTF-8") as f:
                    f.write(self.urdf28(item, out_urdf_dir))
            elif str(item.stem) == "item29":
                with open(urdf_path, mode="w", encoding="UTF-8") as f:
                    f.write(self.urdf29(item, out_urdf_dir))
            elif str(item.stem) == "item30":
                with open(urdf_path, mode="w", encoding="UTF-8") as f:
                    f.write(self.urdf30(item, out_urdf_dir))
            elif str(item.stem) == "item31":
                with open(urdf_path, mode="w", encoding="UTF-8") as f:
                    f.write(self.urdf31(item, out_urdf_dir))
            elif str(item.stem) == "item32":
                with open(urdf_path, mode="w", encoding="UTF-8") as f:
                    f.write(self.urdf32(item, out_urdf_dir))
            elif str(item.stem) == "item33":
                with open(urdf_path, mode="w", encoding="UTF-8") as f:
                    f.write(self.urdf33(item, out_urdf_dir))
            elif str(item.stem) == "item34":
                with open(urdf_path, mode="w", encoding="UTF-8") as f:
                    f.write(self.urdf34(item, out_urdf_dir))
            elif str(item.stem) == "item35":
                with open(urdf_path, mode="w", encoding="UTF-8") as f:
                    f.write(self.urdf35(item, out_urdf_dir))
            elif str(item.stem) == "item36":
                with open(urdf_path, mode="w", encoding="UTF-8") as f:
                    f.write(self.urdf36(item, out_urdf_dir))
            elif str(item.stem) == "item37":
                with open(urdf_path, mode="w", encoding="UTF-8") as f:
                    f.write(self.urdf37(item, out_urdf_dir))
            elif str(item.stem) == "item38":
                with open(urdf_path, mode="w", encoding="UTF-8") as f:
                    f.write(self.urdf38(item, out_urdf_dir))
            elif str(item.stem) == "item39":
                with open(urdf_path, mode="w", encoding="UTF-8") as f:
                    f.write(self.urdf39(item, out_urdf_dir))
            elif str(item.stem) == "item40_1":
                with open(urdf_path, mode="w", encoding="UTF-8") as f:
                    f.write(self.urdf40_1(item, out_urdf_dir))
            elif str(item.stem) == "item40_2":
                with open(urdf_path, mode="w", encoding="UTF-8") as f:
                    f.write(self.urdf40_2(item, out_urdf_dir))

