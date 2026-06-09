@echo off
chcp 65001 >nul 2>&1
echo 正在生成项目报告 docx 文件...
echo.

cd /d "%~dp0"

:: 检查 node 是否可用
where node >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Node.js，请先安装 Node.js 或将其添加到 PATH。
    pause
    exit /b 1
)

:: 检查 docx 模块是否存在
if not exist "node_modules\docx" (
    echo [信息] 正在安装 docx 依赖...
    npm install docx --save
    if errorlevel 1 (
        echo [错误] docx 安装失败。
        pause
        exit /b 1
    )
)

:: 运行生成脚本
node scripts\generate_report.js

echo.
if exist "document\项目报告.docx" (
    echo [完成] 报告已生成: document\项目报告.docx
    echo.
    echo 按任意键打开文件...
    pause >nul
    start "" "document\项目报告.docx"
) else (
    echo [错误] 生成失败，请查看上方错误信息。
    pause
)
