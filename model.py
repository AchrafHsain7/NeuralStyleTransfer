import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#loading the VGG-19 model to use 
image_size = 400
new_image = False

"""
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet'
                                , input_shape= (image_size, image_size, 3))
vgg.save('vgg-model.tf')
"""


#plt.ion()



def show_image(image):
    plt.imshow(image)
    plt.show()


vgg = tf.keras.models.load_model("vgg-model.tf")
vgg.trainable = False

content_image = Image.open("content.jpg")
style_image = Image.open("style.jpg")


#show_image(content_image)
#show_image(style_image)



style_layers = [
    ('block1_conv1', 0.05),
    ('block2_conv1', 0.05),
    ('block3_conv1', 0.2),
    ('block4_conv1', 0.3),
    ('block5_conv1', 0.4)
]

content_layer = [("block5_conv4", 1)]





def compute_content_cost(content_output, generated_output):

    a_C = content_output[-1]
    a_G = generated_output[-1]

    _,n_H,n_W,n_C = a_G.get_shape().as_list()

    a_C_unrolled = tf.transpose(tf.reshape(a_C, shape=[1,-1,n_C]), perm=[0,2,1])
    a_G_unrolled = tf.transpose(tf.reshape(a_G, shape=[1,-1,n_C]), perm=[0,2,1])

    J_content = 1/(4*n_C*n_H*n_W) * tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled , a_G_unrolled)))

    return J_content


def gram_matrix(A):

    GA = tf.linalg.matmul(A, tf.transpose(A))

    return GA


def compute_style_cost_layer(a_S, a_G):

    _,n_H,n_W,n_C = a_G.get_shape().as_list()

    #reshaping to (n_C, n_H*n_W)
    a_S = tf.transpose(tf.reshape(a_S, shape=[-1, n_C]), perm=[1,0])
    a_G = tf.transpose(tf.reshape(a_G, shape=[-1, n_C]), perm=[1,0])

    #creating gram matrixes
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    J_style_layer = tf.multiply(1/(2*n_C*n_H*n_W)**2, tf.reduce_sum(tf.square(tf.subtract(GS, GG))))

    return J_style_layer


def compute_style_cost(style_image_output, generated_image_output, style_layers=style_layers):

    J_style = 0

    a_S = style_image_output[:-1]
    a_G = generated_image_output[:-1]

    for i, weight in zip(range(len(a_S)), style_layers):

        J_style_layer = compute_style_cost_layer(a_S[i], a_G[i])

        J_style +=  weight[1] * J_style_layer

    return J_style






def get_layer_outputs(vgg, layer_names):
    outputs = [vgg.get_layer(layer[0]).output for layer in layer_names]
    model = tf.keras.Model((vgg.inputs), outputs)

    return model


#utilities
def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)

    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]

    return Image.fromarray(tensor)




#main Program
if new_image:
    with open("epoch.txt", "w") as f:
        f.write("0")

with open("epoch.txt", "r") as f:
    epoch = int(f.read())

if new_image:
    content_image = np.array(Image.open("content.jpg").resize((image_size, image_size)))
    content_image = tf.constant(np.reshape(content_image, (1,) + content_image.shape))
else:
    content_image = np.array(Image.open(f"outputs/image_{epoch}.jpg").resize((image_size, image_size)))
    content_image = tf.constant(np.reshape(content_image, (1,) + content_image.shape))

style_image = np.array(Image.open("style.jpg").resize((image_size, image_size)))
style_image = tf.constant(np.reshape(style_image, (1,) + style_image.shape))

generated_image = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32), trainable=True)
noise = tf.random.uniform(tf.shape(generated_image), -0.25, 0.25)
generated_image = tf.add(generated_image, noise)
generated_image = tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0)

plt.imshow(generated_image[0])
plt.show()


#creating the model
vgg_model_outputs = get_layer_outputs(vgg, style_layers + content_layer)

#activation for content image
content_traget = vgg_model_outputs(content_image)
#activation for style image
style_target = vgg_model_outputs(style_image)

#preprocessing style and content and creating a_C, a_S
preprocessed_content = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
a_C = vgg_model_outputs(preprocessed_content)

preprocessed_style = tf.Variable(tf.image.convert_image_dtype(style_image, tf.float32))
a_S = vgg_model_outputs(style_image)

optimizer = tf.keras.optimizers.Adam(0.001)

@tf.function()
def total_cost(J_content, J_style, alpha=10, beta=40):

    J = alpha * J_content + beta * J_style

    return J

@tf.function()
def train_step(generated_image):

    with tf.GradientTape() as tape:

        a_G = vgg_model_outputs(generated_image)

        J_content = compute_content_cost(a_C, a_G)

        J_style = compute_style_cost(a_S, a_G)

        J = total_cost(J_content, J_style, alpha=10, beta=40)

    gradient = tape.gradient(J, generated_image)
    
    optimizer.apply_gradients([(gradient, generated_image)])
    generated_image.assign(clip_0_1(generated_image))

    return J

generated_image = tf.Variable(generated_image)

def create_image(epochs=1001, current_epoch=700):
    
    for i in range(epochs - current_epoch):
        #print(i)
        train_step(generated_image)

        if i % 25 == 0:
            print(f"Epoch:{i}")
        if i % 100 == 0:
            image = tensor_to_image(generated_image)
            #plt.imshow(image)
            image.save(f"outputs/image_{i}.jpg")
            with open("epoch.txt", "w") as f:
                f.write(str(i))
            


create_image(current_epoch=epoch)

# Show the 3 images in a row
fig = plt.figure(figsize=(16, 4))
ax = fig.add_subplot(1, 3, 1)
plt.imshow(content_image[0])
ax.title.set_text('Content image')
ax = fig.add_subplot(1, 3, 2)
plt.imshow(style_image[0])
ax.title.set_text('Style image')
ax = fig.add_subplot(1, 3, 3)
plt.imshow(generated_image[0])
ax.title.set_text('Generated image')
plt.show()


   
#plt.ioff()





