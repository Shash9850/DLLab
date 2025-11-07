base_model = VGG16(weights=weights_path,include_top=False,input_shape=(32,32,3))
for layer in base_model.layers:
    layer.trainable=True

for layer in base_model.layers[:-4]:
    layer.trainable=False
    
base_model.summary()
