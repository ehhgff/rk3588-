#!/bin/sh
#
# Setup chromium environment.
#

# 注意: 此脚本已被修改，默认不在启动时执行
# 如需启动，请运行: /etc/init.d/S99chromium-wayland.sh start
# 或运行: start-optional-services.sh

# 开机自动启动开关 (设置为 true 启用，false 禁用)
AUTO_START=false

case "$1" in
	start)
		# 检查是否允许开机启动
		if [ "$AUTO_START" != "true" ]; then
			echo "S99chromium-wayland.sh: 开机自动启动已禁用 (AUTO_START=false)"
			echo "如需手动启动，请运行: /etc/init.d/S99chromium-wayland.sh start"
			exit 0
		fi
		# Create dummy video node for V4L2 VDA/VEA with rkmpp plugin
		echo dec > /dev/video-dec0
		echo enc > /dev/video-enc0
		;;
	stop)
		;;
	restart|reload)
		;;
	*)
		echo "Usage: $0 {start|stop|restart}"
		exit 1
esac

exit $?
