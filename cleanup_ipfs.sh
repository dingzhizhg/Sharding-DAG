#!/bin/bash

# IPFS 清理脚本
# 用于清理不再需要的IPFS文件，释放存储空间

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}IPFS 存储清理工具${NC}"
echo -e "${BLUE}========================================${NC}\n"

# 检查IPFS是否运行
if ! command -v ipfs &> /dev/null; then
    echo -e "${RED}错误: IPFS 未安装或不在PATH中${NC}"
    exit 1
fi

# 显示当前存储状态
echo -e "${YELLOW}当前IPFS存储状态:${NC}"
ipfs repo stat
echo ""

# 询问清理选项
echo -e "${YELLOW}请选择清理选项:${NC}"
echo "1) 清理未pin的缓存文件（安全，推荐）"
echo "2) 清理所有未pin的文件（包括缓存）"
echo "3) 仅运行垃圾回收（GC）"
echo "4) 显示存储详情"
echo "5) 退出"
read -p "请输入选项 [1-5]: " choice

case $choice in
    1)
        echo -e "\n${YELLOW}正在清理未pin的缓存文件...${NC}"
        # 运行垃圾回收，这会清理未被pin的缓存文件
        ipfs repo gc
        echo -e "${GREEN}✓ 清理完成${NC}"
        ;;
    2)
        echo -e "\n${RED}警告: 这将清理所有未pin的文件！${NC}"
        read -p "确认继续? [y/N]: " confirm
        if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
            echo -e "${YELLOW}正在清理所有未pin的文件...${NC}"
            # 先取消所有pin（除了系统必要的）
            # 然后运行垃圾回收
            ipfs repo gc
            echo -e "${GREEN}✓ 清理完成${NC}"
        else
            echo -e "${YELLOW}操作已取消${NC}"
        fi
        ;;
    3)
        echo -e "\n${YELLOW}正在运行垃圾回收...${NC}"
        ipfs repo gc
        echo -e "${GREEN}✓ 垃圾回收完成${NC}"
        ;;
    4)
        echo -e "\n${YELLOW}存储详情:${NC}"
        echo -e "${BLUE}仓库统计:${NC}"
        ipfs repo stat
        echo ""
        echo -e "${BLUE}Pin的文件数量:${NC}"
        ipfs pin ls --type=recursive | wc -l
        echo ""
        echo -e "${BLUE}最近添加的文件（前10个）:${NC}"
        ipfs refs local | head -10
        # 选项4只显示信息，不执行清理，直接退出
        echo -e "\n${GREEN}========================================${NC}"
        echo -e "${GREEN}信息显示完成！${NC}"
        echo -e "${GREEN}========================================${NC}"
        exit 0
        ;;
    5)
        echo -e "${YELLOW}退出${NC}"
        exit 0
        ;;
    *)
        echo -e "${RED}无效选项${NC}"
        exit 1
        ;;
esac

# 显示清理后的状态（仅对执行了清理操作的选项显示）
if [ "$choice" != "4" ] && [ "$choice" != "5" ]; then
    echo ""
    echo -e "${YELLOW}清理后存储状态:${NC}"
    ipfs repo stat
fi

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}清理完成！${NC}"
echo -e "${GREEN}========================================${NC}"

