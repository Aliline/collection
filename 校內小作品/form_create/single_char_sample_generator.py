from PIL import Image, ImageDraw, ImageFont
from math import ceil
import os
import subprocess
import json
txtpath = "C:/Users/USER/pywork2/ppt_help/word_text.txt"


class Sample_generator():
    def __init__(self, corpus, width, height, fontname, dist_name, fontsize=45) -> None:
        super().__init__()
        self.corpus = corpus
        self.max_width = width
        self.max_height = height
        self.fontsize = fontsize
        self.dist_name = dist_name
        self.output = Image.new("RGB", (width, height), (255, 255, 255))
        self.fnt = ImageFont.truetype(fontname, self.fontsize)
        self.drawer = ImageDraw.Draw(self.output)
        self.labels = list()
        self.img_margin_x = 80
        self.img_margin_y = 80

    def check_x_bound(self, text) -> bool:
        cond_1 = self.img_margin_x + \
            self.drawer.textsize(text, font=self.fnt)[0] < self.max_width
        cond_2 = self.max_width - \
            self.drawer.textsize(text, font=self.fnt)[
                0] > 2 * self.img_margin_x
        return cond_1 and cond_2

    def check_y_bound(self, height) -> bool:
        cond_1 = self.img_margin_y + height < self.max_height
        cond_2 = self.max_height - height > 2 * self.img_margin_y
        return cond_1 and cond_2

    def get_y_position(self, count) -> int:
        return 0 if count == 0 else self.fontsize * count * 1.5

    def write_text(self, y_position, text, pic_num) -> None:
        def get_width(text):
            return self.drawer.textsize(text, font=self.fnt)[0]
        # a = '0 1 2 3 4 5 6 7 8 9 A B C D E F G H I J K L M N O P Q R S T U V W X Y Z'
        # for i in a.split(' '):
        # word = f'U V'
        # a = get_width(word.split(' ')[0])
        # b = get_width(word.split(' ')[1])
        # c = get_width(word)
        # print(f'({word.split(" ")[0]} = {a}, {word.split(" ")[1]} = {b})  =>  {a + b}, {c}  =>  {c - (a + b)}')
        # print(a)
        self.drawer.text((self.img_margin_x, self.img_margin_y +
                          y_position), text, (0, 0, 0), font=self.fnt)
        cul_width = self.img_margin_x
        word_list = text.split(' ')[:-1]
        fontname = self.dist_name.split('.')[1]
        outlier = {
            'MicrosoftJhengHei': {
                'j': 6,
                '/': 10,
                '0': 10,
                '\\': 10,
                '_': 10,
                'd': 10,
                'g': 10,
                'x': 10,
            },
            'MicrosoftJhengHeiBold': {
                ',': 9,
                'S': 10,
                'T': 10,
                'W': 10,
                'A': 9,
                'V': 10,
                'G': 10,
                'Z': 10,
                '\\': 10,
                ']': 10,
                '_': 10,
                'g': 9,
                'j': 7,
                'w': 9,
                'z': 9
            },
            'DFKai-SB': {},
            'MingLiU': {},
            'MicrosoftJhengHeiLight': {
                ']': 12,
                'a': 11,
            },
            'PingFangHeavy': {},
            'PingFangMedium': {},
            'PingFangRegular': {
                'V': 14,
                'X': 14,
                '※': 14,
            },
            'PingFangTCMedium': {
                'V': 14,
            },
            'MicrosoftYaHei': {
                ']': 12,
                'a': 12,
                'g': 12,
                'j': 6,
            }
        }

        def cal_space_size(word):
            if len(word) > 1 and word[0] in outlier[fontname].keys():
                return outlier[fontname][word[0]]
            if word in outlier[fontname].keys():
                return outlier[fontname][word]
            else:
                return get_width(f'1 {word}') - (get_width('1') + get_width(word))

        space_size = {
            'DFKai-SB': lambda x: ceil(self.fontsize / 2),
            'MingLiU': lambda x: ceil(self.fontsize / 2),
            'MicrosoftJhengHei': lambda x: outlier[fontname][x] if x in outlier[fontname].keys() else 11,
            'MicrosoftJhengHeiBold': lambda x: cal_space_size(x),
            'MicrosoftJhengHeiLight': lambda x: outlier[fontname][x] if x in outlier[fontname].keys() else get_width(f'1 {x}') - (get_width('1') + get_width(x)),
            'PingFangHeavy': lambda x: outlier[fontname][x] if x in outlier[fontname].keys() else get_width(f'1 {x}') - (get_width('1') + get_width(x)),
            'PingFangMedium': lambda x: outlier[fontname][x] if x in outlier[fontname].keys() else get_width(f'1 {x}') - (get_width('1') + get_width(x)),
            'PingFangRegular': lambda x: outlier[fontname][x] if x in outlier[fontname].keys() else get_width(f'1 {x}') - (get_width('1') + get_width(x)),
            'PingFangTCMedium': lambda x: outlier[fontname][x] if x in outlier[fontname].keys() else get_width(f'1 {x}') - (get_width('1') + get_width(x)),
            'MicrosoftYaHei': lambda x: outlier[fontname][x] if x in outlier[fontname].keys() else get_width(f'1 {x}') - (get_width('1') + get_width(x)),

        }
        half_shape_size = {
            'DFKai-SB': lambda x: ceil(self.fontsize / 2),
            'MingLiU': lambda x: ceil(self.fontsize / 2),
            'MicrosoftJhengHei': lambda x: self.drawer.textsize(x, font=self.fnt)[0],
            'MicrosoftJhengHeiBold': lambda x: self.drawer.textsize(x, font=self.fnt)[0],
            'MicrosoftJhengHeiLight': lambda x: self.drawer.textsize(x, font=self.fnt)[0],
            'PingFangHeavy': lambda x: self.drawer.textsize(x, font=self.fnt)[0],
            'PingFangMedium': lambda x: self.drawer.textsize(x, font=self.fnt)[0],
            'PingFangRegular': lambda x: self.drawer.textsize(x, font=self.fnt)[0],
            'PingFangTCMedium': lambda x: self.drawer.textsize(x, font=self.fnt)[0],
            'MicrosoftYaHei': lambda x: self.drawer.textsize(x, font=self.fnt)[0],
        }
        full_shape_size = {
            'DFKai-SB': lambda x: self.fontsize,
            'MingLiU': lambda x: self.fontsize,
            'MicrosoftJhengHei': lambda x: self.drawer.textsize(x, font=self.fnt)[0],
            'MicrosoftJhengHeiBold': lambda x: self.drawer.textsize(x, font=self.fnt)[0],
            'MicrosoftJhengHeiLight': lambda x: self.drawer.textsize(x, font=self.fnt)[0],
            'PingFangHeavy': lambda x: self.drawer.textsize(x, font=self.fnt)[0],
            'PingFangMedium': lambda x: self.drawer.textsize(x, font=self.fnt)[0],
            'PingFangRegular': lambda x: self.drawer.textsize(x, font=self.fnt)[0],
            'PingFangTCMedium': lambda x: self.drawer.textsize(x, font=self.fnt)[0],
            'MicrosoftYaHei': lambda x: self.drawer.textsize(x, font=self.fnt)[0],
        }
        for idx, word in enumerate(word_list):

            if len(word) > 1:
                half = list(filter(lambda x: 33 <= ord(x) <= 500, word))
                full = list(filter(lambda x: not (33 <= ord(x) <= 500), word))
                char_width = sum([half_shape_size[fontname](
                    i) for i in half]) + sum([full_shape_size[fontname](i) for i in full])
                char_height = self.drawer.textsize(word, font=self.fnt)[1]
            else:
                char_height = self.drawer.textsize(word, font=self.fnt)[1]
                char_width = half_shape_size[fontname](word) if 33 <= ord(
                    word) <= 500 else full_shape_size[fontname](word)
            self.labels.append(
                f'{word} {cul_width} {int(self.max_height - (self.img_margin_y + y_position + char_height))} {int(cul_width + char_width)} {int(self.max_height - (self.img_margin_y + y_position))} {pic_num}')
            span = space_size[fontname](
                word_list[idx + 1 if idx + 1 != len(word_list) else idx])
            self.labels.append(
                f'\t {int(cul_width + char_width)} {int(self.max_height - (self.img_margin_y + y_position + char_height))} {int(cul_width + char_width + span)} {int(self.max_height - (self.img_margin_y + y_position))} {pic_num}')
            cul_width += char_width + span

    def remove_dir(self, dir_name) -> None:
        for file in os.listdir(dir_name):
            os.remove(f'{dir_name}/{file}')
        os.rmdir(dir_name)

    def text2image(self, ) -> None:
        count = 0
        pic_num = 0
        row_text = f'{self.corpus[0]} '
        cps_size = len(self.corpus)
        if not os.path.exists('%temp_img%'):
            os.mkdir('%temp_img%')
        for i in range(cps_size):
            if i + 1 < cps_size:
                temp = row_text + f'{self.corpus[i + 1]} '
                if self.check_x_bound(temp):
                    row_text += f'{self.corpus[i + 1]} '
                else:
                    y = self.get_y_position(count)
                    if self.check_y_bound(y + 2 * (self.fontsize * 1.5)):
                        self.write_text(
                            y_position=y, text=row_text, pic_num=pic_num)
                        count += 1
                    else:
                        self.write_text(
                            y_position=y, text=row_text, pic_num=pic_num)
                        self.output.save(f'%temp_img%/temp_{pic_num}.png')
                        self.output = Image.new(
                            "RGB", (self.max_width, self.max_height), (255, 255, 255))
                        self.drawer = ImageDraw.Draw(self.output)
                        count = 0
                        pic_num += 1
                    row_text = f'{self.corpus[i + 1]} '
            else:
                y = self.get_y_position(count)
                self.write_text(y_position=y, text=row_text, pic_num=pic_num)
                self.output.save(f'%temp_img%/temp_{pic_num}.png')
        with open(f'{self.dist_name}.box', 'w', encoding='utf-8') as label_file:
            label_file.write('\n'.join(self.labels) + '\n')
            label_file.close()

    def merge_to_tiff(self) -> None:
        filenames = sorted(os.listdir('%temp_img%'), key=lambda x: int(
            x.replace('temp_', '').replace('.png', '')))
        imgs = [Image.open(f'%temp_img%/{filename}') for filename in filenames]
        imgs[0].save(f'{self.dist_name}.tif', compression="tiff_deflate",
                     save_all=True, append_images=imgs[1:])
        self.remove_dir(dir_name='%temp_img%')

    def excuteCommand(self, cmd):
        ex = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        out, err = ex.communicate()
        status = ex.wait()
        return out.decode()

    def correct_box_file(self):
        box_content = []
        with open(f'{self.dist_name}.box', 'r', encoding='utf-8') as box_file:
            box_content = box_file.readlines()
            with open(f'%temp_label%/{self.dist_name}.txt', 'r', encoding='utf-8') as label_file:
                label_content = label_file.readlines()
                for i in range(0, len(box_content), 2):
                    box_content[i] = f'{box_content[i].split("#")[0]}#{label_content[i // 2]}'
                label_file.close()
            box_file.close()
        with open(f'{self.dist_name}.box', 'w', encoding='utf-8') as box_file:
            box_file.write("".join(box_content))
            box_file.close()

    def do(self) -> None:
        self.text2image()
        self.merge_to_tiff()
        print(f'!!! FILE SAVED DONE: {self.dist_name}.tif !!!')


if __name__ == "__main__":
    fonts = [
        ['kaiu.ttf', 'chi_tra.DFKai-SB.exp0'],
        ['mingliu.ttc', 'chi_tra.MingLiU.exp0'],
        ['msyhl.ttc', 'chi_tra.MicrosoftYaHei.exp0'],
        ['msjh.ttc', 'chi_tra.MicrosoftJhengHei.exp0'],
        ['msjhbd.ttc', 'chi_tra.MicrosoftJhengHeiBold.exp0'],
        ['msjhl.ttc', 'chi_tra.MicrosoftJhengHeiLight.exp0'],
        ['PingFang Heavy.ttf', 'chi_tra.PingFangHeavy.exp0'],
        ['PingFang Medium.ttf', 'chi_tra.PingFangMedium.exp0'],
        ['PingFang Regular.ttf', 'chi_tra.PingFangRegular.exp0'],
        ['PingFangTC-Medium.ttf', 'chi_tra.PingFangTCMedium.exp0'],
    ]
    with open(txtpath, 'r', encoding='utf-8') as f:
        corpus = f.read().replace('\n', ' ').split(' ')[:-1]
        for font in fonts:
            generator = Sample_generator(
                width=1800, height=1800, fontname=font[0], corpus=corpus, dist_name=font[1])
            generator.do()
        f.close()
