# -*- coding: UTF-8 -*-
from PIL.Image import NONE
import win32com.client
from win32com.client import DispatchEx
from win32gui import GetTextExtentPoint32, DeleteObject, SelectObject, LOGFONT, CreateDC, CreateCompatibleBitmap, CreateFontIndirect, GetDC
from win32print import GetDeviceCaps
import os
PPTApp = DispatchEx("PowerPoint.Application")
PPTApp.Visible = True    # 为了便于查阅PPT工作情况，这里设置为可见
PPTApp.DisplayAlerts = False   # 为了使工作不中断，忽略可能的弹出警告

output = 'D:/ppt_help/pptx/'
ppt_file = "C:/Users/USER/pywork2/ppt_help/Text_shape.pptx"


def muldiv(num, numerator, denominator):
    k = float(num)
    k = float(k) * numerator
    k = float(k) / denominator
    return int(k)


def getlableSize(text, font, size):
    tempdc = 0
    tempbmp = 0
    f = 0
    lf = LOGFONT()
    textsize = 0
    tempdc = CreateDC("DISPLAY", "", None)
    tempbmp = CreateCompatibleBitmap(tempdc, 1, 1)
    tempobj = SelectObject(tempdc, tempbmp)
    # DeleteObject(tempobj)
    lf.lfFaceName = font
    # lf.lfHeight = muldiv(size, GetDeviceCaps(GetDC(0), 90), 72)
    lf.lfHeight = size
    f = CreateFontIndirect(lf)
    # DeleteObject(SelectObject(tempdc, f))
    tempobj = SelectObject(tempdc, f)
    # DeleteObject(tempobj)
    textsize = GetTextExtentPoint32(tempdc, text)
    return textsize


def RGB(red, green, blue):
    assert 0 <= red <= 255
    assert 0 <= green <= 255
    assert 0 <= blue <= 255
    return red + (green << 8) + (blue << 16)


cx, cy = getlableSize("!", "kaiu", 49)
print("Width={}".format(cx))
print("Height={}".format(cy))


# 抓取正在執行的powerpoint
# PPTApp = win32com.client.GetActiveObject("PowerPoint.Application")
# 抓取正在開啟的簡報
PPTpresentation = PPTApp.Presentations.Open(ppt_file)

# 裡面有幾張簡報?
print("There are {} slides in my presentation".format(
    PPTpresentation.Slides.Count))
# PPTSldRng = PPTpresentation.Slides.Range([1])
PPTSlide1 = PPTpresentation.Slides(1)
# 增加一個TextBox
shape1 = PPTSlide1.Shapes.AddTextbox(
    Orientation=0x1, Left=100, Top=100, Width=100, Height=100)
# 內容
shape1.TextFrame.TextRange.Text = '日月光'
# 大小
shape1.TextFrame.TextRange.Font.Size = 22
# 中文專用字形設定
shape1.TextFrame.TextRange.Font.NameFarEast = "Microsoft YaHei"

shape2 = PPTSlide1.Shapes.AddTextbox(
    Orientation=0x1, Left=100, Top=130, Width=100, Height=100)
shape2.TextFrame.TextRange.Text = '日月光'
shape2.TextFrame.TextRange.Font.Size = 22
shape2.TextFrame.TextRange.Font.NameFarEast = "Microsoft YaHei"
# 粗體
shape2.TextFrame.TextRange.Font.Bold = True

shape3 = PPTSlide1.Shapes.AddTextbox(
    Orientation=0x1, Left=100, Top=160, Width=100, Height=100)
shape3.TextFrame.TextRange.Text = '日月光'
shape3.TextFrame.TextRange.Font.Size = 22
shape3.TextFrame.TextRange.Font.NameFarEast = "Microsoft YaHei"
# 斜體
shape3.TextFrame.TextRange.Font.Italic = True

shape4 = PPTSlide1.Shapes.AddTextbox(
    Orientation=0x1, Left=100, Top=190, Width=100, Height=100)
shape4.TextFrame.TextRange.Text = '日月光'
shape4.TextFrame.TextRange.Font.Size = 22
shape4.TextFrame.TextRange.Font.NameFarEast = "Microsoft YaHei"
# 底線
shape4.TextFrame.TextRange.Font.Underline = True

# 新增一個長方形
shape5 = PPTSlide1.Shapes.AddShape(
    Type=1, Left=100, Top=185, Width=100, Height=100)
shape5.TextFrame2.TextRange.Text = '日月光'
shape5.TextFrame2.TextRange.Font.Size = 22
shape5.TextFrame2.TextRange.Font.NameFarEast = "Microsoft YaHei"
# 只有圖形中的文字才有的刪除號
shape5.TextFrame2.TextRange.Font.Strike = 1
# 設定字形顏色
shape5.TextFrame.TextRange.Font.Color = 0
# 設定靠左對齊
shape5.TextFrame2.TextRange.ParagraphFormat.Alignment = 1
# 長方形設定無填滿
shape5.Fill.Visible = False
# 長方形設定無框線
shape5.Line.Visible = False

shape6 = PPTSlide1.Shapes.AddTextbox(
    Orientation=0x1, Left=100, Top=250, Width=100, Height=100)
shape6.TextFrame.TextRange.Text = '日月光'
shape6.TextFrame.TextRange.Font.Size = 22
shape6.TextFrame.TextRange.Font.NameFarEast = "Microsoft YaHei"
# 底線
shape6.TextFrame.TextRange.Font.Shadow = True

shape7 = PPTSlide1.Shapes.AddShape(
    Type=1, Left=100, Top=245, Width=100, Height=100)
shape7.TextFrame2.TextRange.Text = '日月光'
shape7.TextFrame2.TextRange.Font.Size = 22
shape7.TextFrame2.TextRange.Font.NameFarEast = "Microsoft YaHei"
# 設定字形顏色
shape7.TextFrame.TextRange.Font.Color = 0
shape7.TextFrame.TextRange.Font.Bold = True
# 設定靠左對齊
shape7.TextFrame2.TextRange.ParagraphFormat.Alignment = 1
# 長方形設定無填滿
shape7.Fill.Visible = False
# 長方形設定無框線
shape7.Line.Visible = False

shape7.TextFrame2.TextRange.Characters.Font.Line.Visible = True
shape7.TextFrame2.TextRange.Characters.Font.Fill.ForeColor.RGB = RGB(
    255, 255, 255)

shape8 = PPTSlide1.Shapes.AddShape(
    Type=1, Left=100, Top=275, Width=100, Height=100)
shape8.TextFrame2.TextRange.Text = '日月光'
shape8.TextFrame2.TextRange.Font.Size = 22
shape8.TextFrame2.TextRange.Font.NameFarEast = "Microsoft YaHei"
# 設定字形顏色
shape8.TextFrame.TextRange.Font.Color = 0
shape8.TextFrame.TextRange.Font.Bold = True
shape8.TextFrame.TextRange.Font.Shadow = True
# 設定靠左對齊
shape8.TextFrame2.TextRange.ParagraphFormat.Alignment = 1
# 長方形設定無填滿
shape8.Fill.Visible = False
# 長方形設定無框線
shape8.Line.Visible = False
shape8.TextFrame2.TextRange.Characters.Font.Line.Visible = True
shape8.TextFrame2.TextRange.Characters.Font.Line.ForeColor.RGB = RGB(
    255, 255, 255)
shape8.TextFrame2.TextRange.Characters.Font.Fill.ForeColor.RGB = RGB(
    0, 0, 0)
if(not os.path.isdir(output)):
    os.makedirs(output)


# ppt輸出
# fullpath = os.path.join(output, "test.gif")
# # print(fullpath)
# # PPTpresentation.SaveAs(fullpath)
# # jpg輸出
# PPTSlide1.Export(FileName=fullpath, FilterName="GIF")
# PPTpresentation.close()


# PPTShape1 = PPTSlide1.Shapes(1)
# print(PPTShape1.Name)
# PPTShape1.Select()
# ShpRng = PPTApp.ActiveWindow.Selection.ShapeRange
# print(ShpRng.TextFrame2.TextRange.ParagraphFormat.Alignment)
# ShpRng.Height = 400
# ShpRng.TextRange.Text = "Here is some test text"
