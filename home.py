import os,time
import numpy as np
from keras.models import load_model
import cv2
from flask import Flask,request,render_template,redirect,url_for
from skimage import io
import matplotlib.pyplot as plt


app = Flask(__name__,template_folder='.')
@app.route('/',methods= ['POST','GET'])
def input():
    if request.method == 'POST':
        t = time.time()
        f = request.files['file']
        f.save("static/MinPro/input"+str(t)+".jpg")
        f.close()
        img = cv2.imread("static/MinPro/input"+str(t)+".jpg")
        img = cv2.resize(img,(128,128))
        cv2.imwrite("static/MinPro/input1"+str(t)+".jpg", img)
        cv2.imwrite("static/MinPro/input"+str(t)+".jpg", img)
        img = np.reshape(img, [1, 128, 128, 3])
        base_model = load_model('./custom.h5')
        ans = base_model.predict(img)
        ans = list(ans[0])
        print(ans)
        ans_label = ['glioma_tumor','meningioma_tumor','pituitary_tumor','no_tumor']
        messages = []
        final_ans = ans_label[ans.index(max(ans))]
        ind = 1
        for i,j in zip(ans_label,ans):
            messages.append(str(ind)+'. '+i+' - '+str(j*100)+" %")
            ind+=1
        messages.append("The disease predicted by the model is "+final_ans+" - "+str(max(ans)*100)+" percent")
        if final_ans!='no_tumor':
            local_model = load_model("./seg_model.h5",compile=False)
            X = np.empty((1, 128, 128, 3))
            img = io.imread("static/MinPro/input"+str(t)+".jpg")
            img = cv2.resize(img, (128, 128))
            img = np.array(img, dtype=np.float64)
            img -= img.mean()
            img /= img.std()
            X[0,] = img
            ans_local = local_model.predict(X)
            pred = np.array(ans_local).squeeze().round()
            plt.imsave("static/MinPro/show"+str(t)+".jpg", pred)
            plt.imsave("static/MinPro/show1"+str(t)+".jpg",pred)
            img = cv2.imread("static/MinPro/input"+str(t)+".jpg")
            mask = cv2.imread("static/MinPro/show"+str(t)+".jpg")
            dst = cv2.addWeighted(img, 0.5, mask, 0.5, 0)
            cv2.imwrite('static/MinPro/imposed'+str(t)+'.jpg',dst)
        return render_template("result.html",messages=messages,time=str(t),final_ans=final_ans)
    else:
        return render_template("home.html")

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static',filename='MinPro/'+filename),code=301)


if __name__ == '__main__':
	port = int(os.environ.get('PORT', 9999))
	app.run(host='localhost', port=port)