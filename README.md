# cat-dog-predictor

## Training
### first epoch
Epoch 1/100
150/150 [==============================] - 77s 510ms/step - loss: 0.6970 - acc: 0.5133 - val_loss: 0.6837 - val_acc: 0.5001

I have a loss of 69% and an accuracy of 51%

### fifty epoch
Epoch 50/100
150/150 [==============================] - 84s 558ms/step - loss: 0.4985 - acc: 0.7558 - val_loss: 0.4644 - val_acc: 0.7858

It improved by a lot. I now have a loss of 50% and an accuracy of 75%

### one hundred epoch
Epoch 100/100
150/150 [==============================] - 100s 669ms/step - loss: 0.4207 - acc: 0.8079 - val_loss: 0.3969 - val_acc: 0.8255

By 100 epoch, I now got an accuracy of 80% and a loss of 42%. 80% seemed like the best I could achieve with my model because it hasn't improved since the 70th epoch

## Testing
after running 
```train_img_generator.class_indices```, i got {'CAT': 0, 'DOG': 1}. 
If my predictor outputs a 0, that means the predictor predicted a cat and if it outputs a 1, that would mean it predicted a dog

After creating my cat test image, I asked my model to predict the image.
```model.predict_classes(cat_image)```
The model predicted it correctly as it output array([[0]]) as the result.
```model.predict(cat_image)``` also gave me an output of array([[0.37897196]], dtype=float32).
Anything above .5 means that it predicted a dog and anything below means it predicted a cat
