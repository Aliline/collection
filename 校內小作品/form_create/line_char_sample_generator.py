from PIL import Image, ImageDraw, ImageFont
import os, subprocess, json


class Sample_generator():
    def __init__(self, corpus, width, height, fontname, dist_name, fontsize = 45) -> None:
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
        cond_1 = self.img_margin_x + self.drawer.textsize(text, font = self.fnt)[0] < self.max_width
        cond_2 = self.max_width - self.drawer.textsize(text, font = self.fnt)[0] > 2 * self.img_margin_x
        return cond_1 and cond_2
    
    def check_y_bound(self, height) -> bool:
        cond_1 = self.img_margin_y + height < self.max_height
        cond_2 = self.max_height - height > 2 * self.img_margin_y
        return cond_1 and cond_2
    
    def get_y_position(self, count) -> int:
        return 0 if count == 0 else self.fontsize * count * 1.5

    def write_text(self, y_position, text, pic_num) -> None:
        self.drawer.text((self.img_margin_x, self.img_margin_y + y_position), text, (0, 0, 0), font = self.fnt)
        width, height = self.drawer.textsize(text, font = self.fnt)
        self.labels.append(f'WordStr {self.img_margin_x} {int(self.max_height - (self.img_margin_y + y_position + height))} {int(self.img_margin_x + width)} {int(self.max_height - (self.img_margin_y + y_position))} {pic_num} #{text}')
        self.labels.append(f'\t {int(self.img_margin_x + width + 1)} {int(self.max_height - (self.img_margin_y + y_position + height))} {int(self.img_margin_x + width + 5)} {int(self.max_height - (self.img_margin_y + y_position))} {pic_num}')
    
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
                temp = row_text + f' {self.corpus[i + 1]} '
                if self.check_x_bound(temp):                
                    row_text += f' {self.corpus[i + 1]} '
                else:
                    y = self.get_y_position(count)                    
                    if self.check_y_bound(y + 2 * (self.fontsize * 1.5)):
                        self.write_text(y_position = y, text = row_text, pic_num = pic_num)
                        count += 1
                    else:
                        self.write_text(y_position = y, text = row_text, pic_num = pic_num)
                        self.output.save(f'%temp_img%/temp_{pic_num}.png')
                        self.output = Image.new("RGB", (self.max_width, self.max_height), (255, 255, 255))        
                        self.drawer = ImageDraw.Draw(self.output)
                        count = 0
                        pic_num += 1
                    row_text = f'{self.corpus[i + 1]} '
            else:
                y = self.get_y_position(count)
                self.write_text(y_position = y, text = row_text, pic_num = pic_num)
                self.output.save(f'%temp_img%/temp_{pic_num}.png')
        with open(f'{self.dist_name}_{self.fontsize}.box', 'w', encoding='utf-8') as label_file:
            label_file.write('\n'.join(self.labels) + '\n')
            label_file.close()

    def merge_to_tiff(self) -> None:
        filenames = sorted(os.listdir('%temp_img%'), key = lambda x: int(x.replace('temp_', '').replace('.png', '')))
        imgs = [Image.open(f'%temp_img%/{filename}') for filename in filenames]
        imgs[0].save(f'{self.dist_name}_{self.fontsize}.tif', compression = "tiff_deflate", save_all = True, append_images = imgs[1:])
        self.remove_dir(dir_name = '%temp_img%')
        
    def excuteCommand(self, cmd):
        ex = subprocess.Popen(cmd, stdout = subprocess.PIPE, shell = True)
        out, err  = ex.communicate()
        status = ex.wait()
        return out.decode()
    
    def correct_box_file(self):
        box_content = []
        with open(f'{self.dist_name}_{self.fontsize}.box', 'r', encoding='utf-8') as box_file:
            box_content = box_file.readlines()
            with open(f'%temp_label%/{self.dist_name}.txt', 'r', encoding='utf-8') as label_file:
                label_content = label_file.readlines()
                for i in range(0, len(box_content), 2):
                    box_content[i] = f'{box_content[i].split("#")[0]}#{label_content[i // 2]}'
                label_file.close()
            box_file.close()
        with open(f'{self.dist_name}_{self.fontsize}.box', 'w', encoding='utf-8') as box_file:
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
    with open('base_text.txt', 'r', encoding='utf-8') as f:
    # with open('few.txt', 'r', encoding='utf-8') as f:
        corpus = f.read().replace('\n', ' ').split(' ')[:-1]
        for font in fonts:
            for fontsize in [ 20, 30, 45]:
                generator = Sample_generator(width = fontsize * 40, height = fontsize * 40, fontname = font[0], corpus = corpus, dist_name = font[1], fontsize=fontsize)
                generator.do()
        f.close()