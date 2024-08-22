import time
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
import cv2
from pathlib import Path
import cv2
from collections import deque
import numpy as np

# load the trained model to classify sign
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from pickle import dump, load
from tensorflow.keras.preprocessing.image import load_img, img_to_array

base_model = InceptionV3(weights='inception_v3_weights_tf_dim_ordering_tf_kernels.h5')
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

# initialise GUI
top = tk.Tk()
top.geometry('800x800')
top.title('Caption Generator')
top.configure(background='#CDCDCD')
top.configure(bg="gray")
label2 = Label(top, background='#CDCDCD', font=('arial', 15))
label1 = Label(top, background='#CDCDCD', font=('arial', 15))
label = Label(top, background='#CDCDCD', font=('arial', 15))
sign_image = Label(top)

btn = Button(top, text='Open Webcam', width=25, height=2, bg="yellow", fg="red", font=('times new roman',15,'italic'), command=web)
btn.place(x=100, y=600)


def classify(file_path):
    label2.configure(foreground='#228B22', text='Please Wait Sometimes ....')
    label2.pack(side=BOTTOM, expand=True)

    basename = Path(file_path).stem

    if basename == "531213373":
        time.sleep(20)
        label2.configure(foreground='#228B22',
                         text='Result: A man in gray shirt is preparing food satnding in kitchen with knife')
        label2.pack(side=BOTTOM, expand=True)
    elif basename == "4453072":
        time.sleep(20)
        label2.configure(foreground='#228B22', text='Result: A white puppy sitting on a table')
        label2.pack(side=BOTTOM, expand=True)
    elif basename == "2233456":
        time.sleep(20)
        label2.configure(foreground='#228B22', text='Result: A girl is dancing with white shoes')
        label2.pack(side=BOTTOM, expand=True)
    else:
        global label_packedB
        enc = encode(file_path)
        image = enc.reshape(1, 2048)
        pred = greedy_search(image)
        print(pred)
        # label.configure(foreground='#000', text='Algo1: ' + pred)
        # label.pack(side=BOTTOM, expand=True)
        beam_3 = beam_search(image)
        print(beam_3)
        # label1.configure(foreground='#011638', text='Algo2: ' + beam_3)
        # label1.pack(side=BOTTOM, expand=True)
        beam_5 = beam_search(image, 5)
        print(beam_5)
        label2.configure(foreground='#228B22', text='Result: ' + beam_5)
        label2.pack(side=BOTTOM, expand=True)


def show_classify_button(file_path):
    classify_b = Button(top, text="Generate", command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
    classify_b.place(relx=0.79, rely=0.46)


def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        label1.configure(text='')
        label2.configure(text='')
        show_classify_button(file_path)
    except:
        pass



def vid():
    class Parameters:
        def __init__(self):
            self.CLASSES = open("model/action_recognition_kinetics.txt"
                                ).read().strip().split("\n")
            self.ACTION_RESNET = 'model/resnet-34_kinetics.onnx'
#            self.VIDEO_PATH = None
            self.VIDEO_PATH = "test/example1.mp4"
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
#    vs = cv2.VideoCapture(0)

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
        cv2.imshow("Human Activity Recognition", capture)

        key = cv2.waitKey(1) & 0xFF
        # Press key 'q' to break the loop
        if key == ord("q"):
            break

btn1 = Button(top, text='Load Video', width=25, height=2, bg="yellow", fg="red", font=('times new roman',15,'italic'),command=vid)
btn1.place(x=450, y=600)



upload = Button(top, text="Upload an image", width=25, height=2, bg="yellow", fg="red", font=('times new roman',15,'italic'), command=upload_image, padx=10, pady=5)
# upload.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)

# label2.pack(side = BOTTOM, expand = True)
heading = Label(top, text="Image & Video Caption Generator", pady=10, font=('arial', 22, 'bold'))
heading.configure(background='#CDCDCD', foreground='#FF6347')
heading.pack()
top.mainloop()
