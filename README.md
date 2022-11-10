# COSC576-Project
###### Aesthetic Image Captioning For Photography Critique


**1 Task**

Through this project, we will explore the potential for image captioning models to provide substantive critique of photographers images. Aesthetic Image Captioning (AIC), sometimes also referred to as Aesthetic Critique Captioning, is the process of producing a caption critiquing an image. Where standard Image Captioning will take an image as input, and output a caption describing the contents of that image, AIC will similarly take an image as input, but then produce a caption critiquing the aesthetic quality of that image.  This task can be utilized in applications like aesthetics-based image enhancement and photography cameras.

One of the challenges of this task is the quality of data that the models are to be trained on. When it comes to standard Image Captioning, images can be fairly grounded in an observable ground truth. While a single image may be described in multiple different ways, there is no disputing what is in that image. When it comes to AIC however, people’s subjective critiques are much more visible. One person may like an image for a particular reason, while another person may dislike the photo for that same reason. When there is not enough data to smooth out the variance in these critiques, the dataset may become noisy, causing the model to not perform as well as it could. In this regard, the task at hand will be twofold. Simply developing a powerful model alone will not be sufficient, but a good dataset will need to be used in conjunction with a good model to produce effective results in the field of Aesthetic Image Captioning. 

**2 Literature Search**

The literature search that follows will be separated into 2 major sections. The first section will discuss research in the task of standard Image Captioning and explore some of the standout moments of the field. The second section will discuss research in applying this task to Aesthetic Image Analysis to result in the field of Aesthetic Image Captioning. 

<ins>Image Captioning</ins><br>
One of the papers that pushed the field of Image Captioning forward the most was released in 2015 and called Show and Tell. The authors proposed a model architecture that would use a convolutional neural net (CNN) to encode an image into a vector representation and then feed that encoding into a recurrent neural net (RNN) as a decoder to generate text. This CNN-RNN encoder-decoder model worked very well and ended up becoming the state of the art in Image Captioning at the time of its release. 

A year later in 2016, a new paper called Show, Attend, and Tell added attention to the architecture. Where the previous paper would feed the entire image representation in the first step of the RNN, this new paper used attention to allow the model to focus on more relevant parts of the image each time it is generating a part of the caption. Two types of attention were focused on in this paper, soft attention and hard attention. Soft attention will calculate a probability distribution over the entire image while hard attention will randomly focus on one particular part of an image.

With the success of transformers in neural machine translation and other natural language processing tasks, researchers also started using transformers as encoders and decoders for image captioning. A paper released in 2022 called End-to-End Transformer Based Model for Image Captioning proposed a transformer model which replaced the standard CNN encoder with two transformers– one to extract grid features and the second to extract the relationships between them– and then used another transformer as a decoder to extract word and image grid feature relationships.

<ins>Aesthetic Image Captioning</ins><br>
In 2017, the seminal paper in the field of Aesthetic Image Captioning (AIC) called Aesthetic Critiques Generation for Photos was released and introduced a novel dataset called the Photo Critique Captioning Dataset (PCCD) consisting of ~4K images and 60K captions. One of the problems that comes with AIC is the inherent noisiness of user comments. Typos, off-topic comments, among other inconsistencies make the task particularly difficult. This paper utilized a website where professional photographers would critique users' photos, resulting in less noisy data for models to be trained on. This paper used a modified CNN-LSTM model with an individual LSTM for 3 different aspects of an image- Composition, Color and Lighting, and Subject of Photo.

In 2019, two papers sought to further explore this new field of AIC. The first called Aesthetic Attributes Assessment of Images was released in July, and similarly to the previous paper, generated critiques in regards to 5 different standard aesthetic attributes in an image- Color and Lighting; Composition; Depth and Focus; Impression and Subject; and Use of Camera. They developed a dataset called DPC-Captions which contained ~154K images and ~2.4 million captions with mentions of the previous five attributes. It should be noted that this process of using multiple LSTMs trained on each attribute becomes fairly computationally expensive.

The second paper called Aesthetic Image Captioning From Weakly-Labelled Photographs was released just a month later in October and the author's goals were to mitigate the previously mentioned noisiness of user comments, and utilize weakly labeled images in a standard CNN-LSTM model to achieve effective results. Using a probabilistic n-gram filtering strategy, this paper developed a new dataset called AVA-Captions which consisted of ~230K images and 1.1 million filtered and cleaned comments from a previous dataset called AVA-Comments, which was a collection of images and comments retrieved from an online community of amature photographers. 

The most recent paper called Understanding Aesthetics with Language and was released in 2022 and proposed a novel dataset called the Reddit Photo Critique Dataset (RPCD). It contains ~70K images and ~220K filtered and cleaned comments retrieved from a photo critiquing subreddit. The authors attempted to use the dataset in the task of AIC by utilizing a model called BLIP. However, they mentioned that it performed significantly worse on Aesthetic Critique Captioning than it did on standard Image Captioning datasets and that it should be an area for further research in the future.

**3 Data**

In our task, we will consider multiple datasets which contain images and an arbitrary number of caption pairs corresponding to each image. As the datasets vary in both size and quality/noisiness of the data, they are likely to have an impact on model performance. It will therefore be useful to try multiple to allow us to better train and tune a model that can better generalize to the task of Aesthetic Image Captioning.

One dataset that we will use is the Photo Critique Captioning Dataset (PCCD) released along with the 2017 seminal paper in Aesthetic Image Captioning. This dataset contains pairwise data of images and critiques on different aspects of ~4K images and over 60K comments. As the author’s mention in the paper that it was the first publicly available dataset in the photo aesthetics captioning field and as it is constructed from professional photographers comments on users photos, it should be fairly free of noise.

The next dataset that we will try out is the AVA-Captions dataset. AVA-Captions dataset where each image is accompanied by a set of captions that correlates to its visual features. This dataset contains ~230K images with around 5 captions per image for a total of 1.15 million captions. The benefit of using the AVA-Captions dataset is that it is significantly larger than the PCCD dataset, allowing the model to train on more data and potentially become more effective at the task. Compared with AVA-Comments which came before and contained over 1.5 million user comments, AVA-Captions is filtered such that only the most informative comments remain; again limiting the noisiness of the dataset.

The final and potentially one of the most promising datasets we will try is the Reddit Photo Critique Dataset (RPCD) which contains ~74K images and ~220K comments. Compared with previous datasets like AVA and its successors, this dataset generally contains higher definition photos as well as longer and more informative comments.

**4 Strategy**

For this task, we need our training data to be such that each image is associated with a set of comments critiquing the image. Tasks mentioned in the literature create or use existing datasets where there is a strong correlation between the visual features of a given image and its accompanying comments. The existing datasets have been preprocessed and cleaned of noisy comments. Moreover, they have been formatted in such a way that each image is associated with k comments. Hence, we will leverage those datasets.

Previous Image Aesthetic Captioning tasks have relied on a CNN-RNN/LSTM combination to train a model that outputs aesthetic critics of an image or some score related to the image's aesthetic attributes. For training, a CNN is used to extract image features which together with the accompanying comments are fed into the LSTM to predict the image aesthetic caption/critique. 

We plan to use a vision transformer (ViT) instead of a CNN to extract the image visual features which together with the accompanying comments will be fed into BERT to predict the image aesthetic caption/critique. We opt for BERT over the traditional RNN/LSTM model due to its bidirectional and self-attention mechanism. Self-attention refers to the model’s ability to rely on the global input to predict some unseen item. For example, to predict some characters in the middle of a sentence, BERT relies on all characters in the sentence. Due to its attention mechanism, BERT performs better than the sequential RNN/LSTM. 

CNN does not encode the relative position of different features when small filters are used. On the other hand, ViT split the image into a set of patches encoded with their relative position. These patches are then fed into a self-attention module that outputs visual features, hence, ViT learns both features and their relative position to each other. The major difference between ViT and CNN is that the former captures long-range dependencies between features when trained on a lot of data while the latter relies on its strong inductive bias. It has been shown that ViT outperforms CNN 4 times in terms of accuracy and efficiency. 

Note that ViT requires training on large datasets to outperform CNN. Since we do not know yet how much data will be needed to reach our desired accuracy, we will experiment with a mix of CNN and ViT aka ConViT which uses convolutions in early layers and self-attention in later layers.
 
**5 Team Members**

There will be three of us working together on this project: Didier Akilimali, Hanouf Aljlayl, and Keenan Samway.
