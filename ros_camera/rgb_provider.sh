#!/bin/bash
SCRIPT_PATH=`realpath $0`
DIR_PATH=$(dirname $SCRIPT_PATH)
XDG_VTNR=7 LC_PAPER=lzh_TW LC_ADDRESS=lzh_TW LC_MONETARY=lzh_TW ROS_ROOT=/opt/ros/kinetic/share/ros ROS_PACKAGE_PATH=/home/test/ur3_driver/src:/home/test/realsense/src:/opt/ros/kinetic/share ROS_MASTER_URI=http://localhost:11311 LC_NUMERIC=lzh_TW ROS_PYTHON_VERSION=2 ROS_VERSION=1 LD_LIBRARY_PATH=/home/test/ur3_driver/devel/lib:/home/test/realsense/devel/lib:/opt/ros/kinetic/lib:/opt/ros/kinetic/lib/x86_64-linux-gnu LC_TELEPHONE=lzh_TW VIRTUAL_ENV=/home/test/pyenv_pyrobot PATH=/home/test/pyenv_pyrobot/bin:/opt/ros/kinetic/bin:/usr/local/cuda-9.0/bin:/home/test/bin:/home/test/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin LC_IDENTIFICATION=lzh_TW LANG=en_US.UTF-8 GDM_LANG=en_US LC_MEASUREMENT=lzh_TW ROS_DISTRO=kinetic LANGUAGE=en_US PYTHONPATH=/home/test/ur3_driver/devel/lib/python2.7/dist-packages:/home/test/realsense/devel/lib/python2.7/dist-packages:/opt/ros/kinetic/lib/python2.7/dist-packages PKG_CONFIG_PATH=/home/test/ur3_driver/devel/lib/pkgconfig:/home/test/realsense/devel/lib/pkgconfig:/opt/ros/kinetic/lib/pkgconfig:/opt/ros/kinetic/lib/x86_64-linux-gnu/pkgconfig CMAKE_PREFIX_PATH=/home/test/ur3_driver/devel:/home/test/realsense/devel:/opt/ros/kinetic LC_TIME=lzh_TW ROS_ETC_DIR=/opt/ros/kinetic/etc/ros LC_NAME=lzh_TW ${DIR_PATH}/rgb_provider.py
#source ~/.bashrc
#shopt -s expand_aliases
#ur3env
#pyrobotenv
#${DIR_PATH}/rgb_provider.py
