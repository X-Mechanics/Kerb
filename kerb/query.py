from .text import twitter
from .images import load_CIFAR_batch

import streamlit as st
import numpy as np
import pandas as pd


def kerb(text):

    parsed = text.split(' ')
    k = 100  # limit 100
    for p in parsed:
        if p.isdigit():
            k = int(p)

    if 'twitter' in text:
        tweets = twitter()
        if 'positive' in text:
            st.write(list(tweets[tweets['label']  == 1]['tweet'].values)[:k])
        else:
            st.write(list(tweets[tweets['label']  == 0]['tweet'].values)[:k])

    if 'cifar' in text:
        X, y = load_CIFAR_batch('./data/cifar')

        if 'cars' in text:
            select_y = (y == 1)  # 1 represents cars
        if 'ships' in text:
            select_y = (y == 8)  # https://towardsdatascience.com/cifar-10-image-classification-in-tensorflow-5b501f7dc77c

        X = X.astype('float32')
        X_cars = X[select_y]
        cars = []
        for car in X_cars[:50]:
            sample = car / 255
            sample = sample.reshape(3, 32, 32).transpose(1, 2, 0)

            indicator = np.argmax((np.mean(sample[:,:,0]), np.mean(sample[:,:,1]), np.mean(sample[:,:,2])))

            if indicator == 0:
                cars.append(sample)
        st.image(cars[:k], width=100)