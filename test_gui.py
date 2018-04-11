from tkinter import Tk, filedialog, Message
from evaluate import get_model, process_one_file
from tkinter import ttk, Menu
from tkinter.filedialog import askopenfilename
import cv2
from PIL import ImageTk, Image
from process_raw_data import get_crop, resize


model = get_model()


class MyApp():
    def open_file(self):
        name = askopenfilename(title="Choose a file.")
        print('Parsing', name)
        left_image = ImageTk.PhotoImage(Image.fromarray(resize(name, 400)))
        self.ipanel1.configure(image=left_image)
        self.ipanel1.image = left_image

        right_image = ImageTk.PhotoImage(Image.fromarray(get_crop(name, 200)))
        self.ipanel2.configure(image=right_image)
        self.ipanel2.image = right_image

        process_file = 'to_process512x512.png'
        get_crop(name, save_to=process_file)
        best_label, results = process_one_file(process_file, model)
        print(best_label, results)

        _text = ' '.join(('Best match:', best_label))
        self.result_label.configure(text=_text)
        self.result_label.text = _text

        ordered_list = list(results.items())
        ordered_list.sort(key=lambda x: -x[1])

        _text = '\n'.join('{} - {}%'.format(k, str(round(v*100, 2))) for k, v in ordered_list)
        self.results_label.configure(text=_text)
        self.results_label.text = _text

    def __init__(self):
        font_style = ("Helvetica", 16)

        self.root = Tk()

        self.root.geometry("900x700")

        self.label = ttk.Label(self.root, text="What camera made the picture?",
                               foreground="black", font=font_style)
        self.label.grid(row=0, column=0)

        self.crop_label = ttk.Label(self.root, text='NN uses crop 512x512', font=font_style)
        self.crop_label.grid(row=0, column=2, pady=(10,0))

        img1 = ImageTk.PhotoImage(Image.fromarray(resize('iphone6.png', 400)))
        self.ipanel1 = ttk.Label(self.root, image=img1)
        self.ipanel1.grid(row=1, column=0, padx=(10, 10))

        self.help_text = ttk.Label(self.root, text='==================>')
        self.help_text.grid(row=1, column=1)

        img2 = ImageTk.PhotoImage(Image.fromarray(get_crop('iphone6.png', 200)))
        self.ipanel2 = ttk.Label(self.root, image=img2)
        self.ipanel2.grid(row=1, column=2, padx=(10, 10))

        self.result_label = ttk.Label(self.root, text='Results:', font=font_style)
        self.result_label.grid(row=2, column=0)

        self.results_label = Message(self.root, text='', font=("Helvetica", 10))
        self.results_label.grid(row=3, column=0)

        self.menu = Menu(self.root)
        self.root.config(menu=self.menu)

        file = Menu(self.menu)

        file.add_command(label='Open', command=self.open_file)
        file.add_command(label='Exit', command=lambda: exit())

        self.menu.add_cascade(label='File', menu=file)

        self.root.mainloop()


app = MyApp()
app.root.mainloop()