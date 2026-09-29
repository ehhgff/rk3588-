#!/bin/sh
### BEGIN INIT INFO
# Provides:       wifibt-init
# Required-Start: $local_fs $syslog
# Required-Stop:  $local_fs
# Default-Start:  S
# Default-Stop:   K
# Description:    Init Rockchip Wifi/BT
### END INIT INFO

# 注意: 此脚本已被修改，默认不在启动时执行
# 如需启动，请运行: /etc/init.d/S36wifibt-init.sh start
# 或运行: start-optional-services.sh

# 开机自动启动开关 (设置为 true 启用，false 禁用)
AUTO_START=false

case "$1" in
	start)
		# 检查是否允许开机启动
		if [ "$AUTO_START" != "true" ]; then
			echo "S36wifibt-init.sh: 开机自动启动已禁用 (AUTO_START=false)"
			echo "如需手动启动，请运行: /etc/init.d/S36wifibt-init.sh start"
			exit 0
		fi
		/usr/bin/wifibt-init.sh start
		;;
	stop|restart)
		/usr/bin/wifibt-init.sh $1
		;;
	*)
		echo "Usage: [start|stop|restart]" >&2
		exit 3
		;;
esac

:
