from tkinter import Frame, Label

from PIL import Image, ImageTk

from options import Options


class CNNInfoFrame:

    def __init__(self, parent: Frame, dataSetDisplayName: str, images: dict):
        self.frame = Frame(parent)
        Options.instance.addFrame(self.frame)
        self.frame.grid(row=0, column=1)
        Label(self.frame, text=dataSetDisplayName, font=("Helvetica", 32)).grid(row=0, column=0, columnspan=99)
        Label(self.frame, font=("Helvetica", 32)).grid(row=1, column=0, columnspan=99)
        i = 0
        self.photos = []
        total_samples = sum(len(lst) for lst in images.values())
        for key, value in images.items():
            self.frame.columnconfigure(i, weight=1)
            percent = format(len(value) / total_samples * 100, '.1f') + "%"
            Label(self.frame, text="Category \"" + key + "\"").grid(row=2, column=i)
            Label(self.frame, text="Total samples: " + str(len(value)) + " (" + percent + ")").grid(row=3, column=i)
            Label(self.frame, text="Example:").grid(row=4, column=i)

            im = ImageTk.PhotoImage(Image.open(value[0]))
            self.photos.append(im)
            Label(self.frame, image=im).grid(row=5, column=i)
            i += 1

    def getFrame(self):
        return self.frame
