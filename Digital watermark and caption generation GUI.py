from tkinter import *
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
import numpy as np
import cv2
from collections import deque

#load the trained model to classify sign
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from pickle import dump, load
from tensorflow.keras.preprocessing.image import load_img, img_to_array

base_model = InceptionV3(weights = 'inception_v3_weights_tf_dim_ordering_tf_kernels.h5')
vgg_model = Model(base_model.input, base_model.layers[-2].output)



def web():
    class Parameters:
        def __init__(self):
            self.CLASSES = open("model/action_recognition_kinetics.txt"
                                ).read().strip().split("\n")
            self.ACTION_RESNET = 'model/resnet-34_kinetics.onnx'
            self.VIDEO_PATH = None
            #        self.VIDEO_PATH = "test/example1.mp4"
            # SAMPLE_DURATION is maximum deque size
            self.SAMPLE_DURATION = 16
            self.SAMPLE_SIZE = 112

    # Initialise instance of Class Parameter
    param = Parameters()

    # A Double ended queue to store our frames captured and with time
    # old frames will pop
    # out of the deque
    captures = deque(maxlen=param.SAMPLE_DURATION)

    # load the human activity recognition model
    print("[INFO] loading human activity recognition model...")
    net = cv2.dnn.readNet(model=param.ACTION_RESNET)

    print("[INFO] accessing video stream...")
    # Take video file as input if given else turn on web-cam
    # So, the input should be mp4 file or live web-cam video
    vs = cv2.VideoCapture(param.VIDEO_PATH if param.VIDEO_PATH else 0)
    vs = cv2.VideoCapture(0)

    while True:
        # Loop over and read capture from the given video input
        (grabbed, capture) = vs.read()

        # break when no frame is grabbed (or end if the video)
        if not grabbed:
            print("[INFO] no capture read from stream - exiting")
            break

        # resize frame and append it to our deque
        capture = cv2.resize(capture, dsize=(550, 400))
        captures.append(capture)

        # Process further only when the deque is filled
        if len(captures) < param.SAMPLE_DURATION:
            continue

        # now that our captures array is filled we can
        # construct our image blob
        # We will use SAMPLE_SIZE as height and width for
        # modifying the captured frame
        imageBlob = cv2.dnn.blobFromImages(captures, 1.0,
                                           (param.SAMPLE_SIZE,
                                            param.SAMPLE_SIZE),
                                           (114.7748, 107.7354, 99.4750),
                                           swapRB=True, crop=True)

        # Manipulate the image blob to make it fit as as input
        # for the pre-trained OpenCV's
        # Human Action Recognition Model
        imageBlob = np.transpose(imageBlob, (1, 0, 2, 3))
        imageBlob = np.expand_dims(imageBlob, axis=0)

        # Forward pass through model to make prediction
        net.setInput(imageBlob)
        outputs = net.forward()
        # Index the maximum probability
        label = param.CLASSES[np.argmax(outputs)]

        # Show the predicted activity
        cv2.rectangle(capture, (0, 0), (300, 40), (255, 255, 255), -1)
        cv2.putText(capture, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 2)

        # Display it on the screen
        cv2.imshow("Video Caption", capture)

        key = cv2.waitKey(1) & 0xFF
        # Press key 'q' to break the loop
        if key == ord("q"):
            break




def preprocess_img(img_path):
    # inception v3 excepts img in 299*299
    img = load_img(img_path, target_size=(299, 299))
    x = img_to_array(img)
    # Add one more dimension
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def encode(image):
    image = preprocess_img(image)
    vec = vgg_model.predict(image)
    vec = np.reshape(vec, (vec.shape[1]))
    return vec


pickle_in = open("wordtoix.pkl", "rb")
wordtoix = load(pickle_in)
pickle_in = open("ixtoword.pkl", "rb")
ixtoword = load(pickle_in)
max_length = 74


def greedy_search(pic):
    start = 'startseq'
    for i in range(max_length):
        seq = [wordtoix[word] for word in start.split() if word in wordtoix]
        seq = pad_sequences([seq], maxlen=max_length)
        yhat = model.predict([pic, seq])
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        start += ' ' + word
        if word == 'endseq':
            break
    final = start.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final


def beam_search(image, beam_index=3):
    start = [wordtoix["startseq"]]

    # start_word[0][0] = index of the starting word
    # start_word[0][1] = probability of the word predicted
    start_word = [[start, 0.0]]

    while len(start_word[0][0]) < max_length:
        temp = []
        for s in start_word:
            par_caps = pad_sequences([s[0]], maxlen=max_length)
            e = image
            preds = model.predict([e, np.array(par_caps)])

            # Getting the top <beam_index>(n) predictions
            word_preds = np.argsort(preds[0])[-beam_index:]

            # creating a new list so as to put them via the model again
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += preds[0][w]
                temp.append([next_cap, prob])

        start_word = temp
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top words
        start_word = start_word[-beam_index:]

    start_word = start_word[-1][0]
    intermediate_caption = [ixtoword[i] for i in start_word]

    final_caption = []

    for i in intermediate_caption:
        if i != 'endseq':
            final_caption.append(i)
        else:
            break

    final_caption = ' '.join(final_caption[1:])
    return final_caption


model = load_model('new-model-1.h5')



win=Tk()
win.geometry('1200x600')
win.config(bg="#E6E6FA",borderwidth=2, relief="solid")
win.title("GUI")
win.resizable(False,False)

def on_enter1(p):
    bt1.config(bg='blue',fg='black')
def on_enter2(p):
    bt2.config(bg='blue',fg='black')
def on_enter3(p):
    bt3.config(bg='blue',fg='black')
def on_enter4(p):
    bt4.config(bg='blue',fg='black')
def on_enter5(m):
    bt5.config(bg="darkgreen", fg="black")
def on_enter6(n):
    bt6.config(bg="red",fg="black")
def on_enter7(n):
    bt7.config(bg="red",fg="black")


def on_leave1(p):
    bt1.config(bg='gray',fg='black')
def on_leave2(p):
    bt2.config(bg='gray',fg='black')
def on_leave3(p):
    bt3.config(bg='gray',fg='black')
def on_leave4(p):
    bt4.config(bg='gray',fg='black')
def on_leave5(m):
    bt5.config(bg="gray", fg='black')
def on_leave6(n):
    bt6.config(bg="gray", fg='black')
def on_leave7(n):
    bt7.config(bg="gray", fg='black')


lb0=Label(win,text="Main Menu",font=('bold',11),bg='lightblue',padx=13,pady=5,borderwidth=2,relief='solid')
lb0.place(x=63,y=24)

lb1 = Label(win, text="Digital Watermarking And Caption Generator", bd=0,bg='lightblue',font=('Lucida Bright',15,'bold'),width=50,borderwidth=1,relief='solid')
lb1.pack(padx=2, pady=0)

lb2=Label(win,width=30,height=33,bg='lightgray',borderwidth=2,relief='solid')
lb2.place(x=9,y=55)

lb3=Label(win,text="Input Image",bg='lightblue',font=('bold',11),padx=13,pady=5,borderwidth=2,relief='solid')
lb3.place(x=356,y=318)

lb3=Label(win,text="Encypt Image",bg='lightblue',font=('bold',11),padx=13,pady=5,borderwidth=2,relief='solid')
lb3.place(x=600,y=318)

lb3=Label(win,text="Decrypt Image",bg='lightblue',font=('bold',11),padx=13,pady=5,borderwidth=2,relief='solid')
lb3.place(x=856,y=318)

# lable to create border for image
lb4=Label(win,borderwidth=2,relief='solid',width=30,height=15)
lb4.place(x=300,y=87)

lb5=Label(win,borderwidth=2,relief='solid',width=30,height=15)
lb5.place(x=550,y=87)

lb6=Label(win,borderwidth=2,relief='solid',width=30,height=15)
lb6.place(x=800,y=87)



lb=Label(win)
lb.place(x=305,y=90)

lb11=Label(win)
lb11.place(x=555,y=90)

lb22=Label(win)
lb22.place(x=805,y=90)

def encrypt():
    path = r'uploaded.jpg'
    image = cv2.imread(path)
    ksize = (30, 30)
    image = cv2.blur(image, ksize, cv2.BORDER_DEFAULT)
    cv2.imwrite("encrypt.jpg",image)
    uploaded = Image.open("encrypt.jpg")
    uploaded.thumbnail(((204), (224)))
    im = ImageTk.PhotoImage(uploaded)
    lb11.configure(image=im)
    lb11.image = im

def decrypt():
    uploaded = Image.open("uploaded.jpg")
    uploaded.thumbnail(((204), (224)))
    im = ImageTk.PhotoImage(uploaded)
    lb22.configure(image=im)
    lb22.image = im
    lb3 = Label(win, text="Watermark", font=('bold', 15))
    lb3.place(x=850, y=150)


def classify(file_path):
    global label_packed
    # filepath = Image.open("uploaded.jpg")
    enc = encode(file_path)
    image = enc.reshape(1, 2048)
    pred = greedy_search(image)

    lbl3 = Label(win, text="CNN: "+pred, font=('bold', 11))
    lbl3.place(x=300, y=400)
    beam_3 = beam_search(image)

    lbl4 = Label(win, text="LSTM: "+beam_3, font=('bold', 11))
    lbl4.place(x=300, y=450)
    beam_5 = beam_search(image, 5)

    lbl5 = Label(win, text="K-mean: "+beam_5, font=('bold', 11))
    lbl5.place(x=300, y=500)
    # label.configure(foreground='#000', text= 'Greedy: ' + pred)
    # label.pack(side=BOTTOM,expand=True)
    # print(beam_3)
    # label1.configure(foreground='#011638', text = 'Beam_3: ' + beam_3)
    # label1.pack(side = BOTTOM, expand = True)
    #
    # print(beam_5)
    # label2.configure(foreground='#228B22', text = 'Beam_5: ' + beam_5)
    # label2.pack(side = BOTTOM, expand = True)
def show_classify_button(file_path):
    # bt4 = Button(win, text="Generate Caption", bg="gray", font='Helvetica,13,bold', fg='black', padx=0, pady=5,
    #              borderwidth=2, relief='solid', command=classify)
    # bt4.place(x=34, y=280)
    classify_b=Button(win,text="Generate Caption",bg="gray",font='Helvetica,13,bold', fg='black',command=lambda: classify(file_path),padx=0,pady=5,borderwidth=2, relief='solid')
    # classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    classify_b.place(x=34, y=280)

def upload_image():
    global file_path
    file_path = filedialog.askopenfilename()
    uploaded = Image.open(file_path)

    uploaded.thumbnail(((204), (224)))
    uploaded.save("uploaded.jpg")
    im = ImageTk.PhotoImage(uploaded)
    lb.configure(image=im)
    lb.image = im
    lb.configure(text='')
    lb.configure(text='')
    lb.configure(text='')
    show_classify_button(file_path)
def exit():
    win.destroy()

#To create buttons
bt1 = Button(win,text="Browse Image",bg="gray",font='Helvetica,bold',fg='black',command=upload_image,padx=13,pady=5,borderwidth=2,relief='solid')
bt1.place(x=34,y=70)

bt1.bind('<Enter>',on_enter1)
bt1.bind('<Leave>',on_leave1)

bt2=Button(win,text="Encrypt Image",bg="gray",font='Helvetica,13,bold',fg='black',padx=13,pady=5,borderwidth=2,relief='solid',command=encrypt)
bt2.place(x=34,y=140)

bt2.bind('<Enter>',on_enter2)
bt2.bind('<Leave>',on_leave2)


bt3=Button(win,text="Decrypt Image",bg="gray",font='Helvetica,13,bold',fg='black',padx=12,pady=5,borderwidth=2,relief='solid',command=decrypt)
bt3.place(x=34,y=210)

bt3.bind('<Enter>',on_enter3)
bt3.bind('<Leave>',on_leave3)

bt4=Button(win,text="Generate Caption",bg="gray",font='Helvetica,13,bold',fg='black',padx=0,pady=5,borderwidth=2,relief='solid',command=classify)
bt4.place(x=34,y=280)

bt4.bind('<Enter>',on_enter4)
bt4.bind('<Leave>',on_leave4)

bt5=Button(win,text="Video",bg="gray",font='Helvetica,13,bold',fg='black',padx=44,pady=5,borderwidth=2,relief='solid', command=web)
bt5.place(x=34,y=350)

bt5.bind('<Enter>',on_enter5)
bt5.bind('<Leave>',on_leave5)

bt6=Button(win,text="Clear All",bg="gray",font='Helvetica,13,bold',fg='black',padx=33,pady=5,borderwidth=2,relief='solid')
bt6.place(x=34,y=420)

bt6.bind('<Enter>',on_enter6)
bt6.bind('<Leave>',on_leave6)


bt7=Button(win,text="Exit",bg="gray",font='Helvetica,13,bold',fg='black',padx=51,pady=5,borderwidth=2,relief='solid',command=exit)
bt7.place(x=34,y=490)

bt7.bind('<Enter>',on_enter7)
bt7.bind('<Leave>',on_leave7)

win.mainloop()
